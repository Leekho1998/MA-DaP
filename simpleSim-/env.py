# env.py
import gym
from gym import spaces
import numpy as np
from sort_algo import short_duration_first_deploy
from task_host import Task
import logging
from util.reward_func import RewardFunc
from CNN1D import CNN1DRegressor
import torch
import random
import pandas as pd
from util.utils import deal_with_affi, deal_with_resource, features_normalize_func

# 2处归一化：env那里和奖励那里
max_total_energy = 600000 # 用来归一化
max_avg_energy = 5000 # 用来归一化
max_run_time = 100 # 用来归一化  200
max_avg_run_time = 30 # 用来归一化 30
max_avg_wait_time = 10 # 用来归一化 30

random.seed(42)

class SchedulingEnv(gym.Env):
    def __init__(self, tasks, hosts, reward_strategy='my_reward'):
        super(SchedulingEnv, self).__init__()
        self._started_tasks = []  # 本 tick 刚开始运行的任务（用于事件结算）
        self.totalEnergyPrices = 0
        self.energyPricePerHost = np.zeros(len(hosts))
        self.accumulatedCostPerHost = np.zeros(len(hosts))
        self.accumulateEnergyConsumptionPerHost = np.zeros(len(hosts))
        self.energyConsumptionPerHost = np.zeros(len(hosts))
        self.oldEnergyConsumptionPerHost = np.zeros(len(hosts))
        self.tasks = tasks
        self.hosts = hosts
        self.task_dict = {task.task_name: task for task in self.tasks}

        self.maxResource = np.array([0.0,0.0,0.0])
        for host in hosts:
            self.maxResource += host.max_resource


        self.reward_Func = RewardFunc()
        self.reward_strategy = reward_strategy

        self.maxPrice = 10 * 5

        self.current_time = 0

        self.timeQue = np.zeros((144,3))

        self.done = False
        self.total_energy_consumption = 0.0

        self.delay_matrix = np.zeros((len(hosts), len(hosts)))  # 延迟矩阵
        # 加载延迟矩阵
        # self.loss = 0
        # self.delay_matrix_file = f"./dataset/delay_matrix_loss={int(self.loss*10)}.csv"
        # self.delay_matrix = pd.read_csv(self.delay_matrix_file, index_col=0)
        # print(f"check delay[0][1]: {self.delay_matrix.iloc[0, 1]}") 
        self.bw = 1000  # 带宽(1000Mbps)

        # self.action_space = spaces.Discrete(len(hosts))  # 动作是选择主机
        # self.observation_space = spaces.Box(low=0, high=1, shape=(len(tasks),), dtype=np.float32)  # 状态空间
        # 加载预训练的1DCNN模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.energy_model = CNN1DRegressor().to(self.device)
        self.energy_model.load_state_dict(torch.load("./saved_model/power_model/cnn1d_model.pth", map_location=self.device))
        self.energy_model.eval()

        self.best_F = 1000
        self.best_step = 0
        self.best_avg_wait_time = 0
        self.best_avg_run_time = 0
        self.best_energy = 0

        # 初始化每个主机的 env 属性
        for host in self.hosts:
            host.env = self

    def reset(self):
        # 重置环境
        for host in self.hosts:
            host.reset()
        for task in self.tasks:
            task.reset()
        self.current_time = 0
        self.total_energy_consumption = 0.0
        self.totalEnergyPrices = 0.0
        self.energyConsumptionPerHost = np.zeros(len(self.hosts))
        self.accumulateEnergyConsumptionPerHost = np.zeros(len(self.hosts))
        self.accumulatedCostPerHost = np.zeros(len(self.hosts))
        self.energyPricePerHost = np.zeros(len(self.hosts))
        self.done = False
        self.reward_Func.reset()
        # 返回初始状态
        return self.get_no_task_state(self.hosts, [1]*len(self.hosts))
    
    def set_reward_params(self, w1, w2):
        self.reward_Func.set_params(w1, w2)
    
    def normal_step(self):
        self.execute_tasks() # 只是时间步往前推进一步
        completed_tasks = sum(task.is_completed(self.current_time) for task in self.tasks)
        print(f"Curtime: {self.current_time}, Completed tasks: {completed_tasks}, Total tasks: {Task.total_tasks}")
        done = completed_tasks == Task.total_tasks   # total_tasks是所有任务的总数300

        logging.info(f"Current time: {self.current_time}, Done: {completed_tasks}")

        self.print_hosts_info()
        reward = 0
        state = self.get_no_task_state(self.hosts, [1]*len(self.hosts))
        info = {}
        return state, reward, done, info  # 这个函数的这四个值不会影响任何东西，但我还是返回了，相当于返回了一些可供参考的信息吧

    def schedule_step(self, task, chosen_host, host_mask, action):
        # len(tasks_to_schedule) > 0才会进入,只会被强化学习算法调用
        if action == 40:
            assign_flag = False
        else :
            assign_flag = chosen_host.placement_possible(task)
        if assign_flag: # 如果可以分配
            self.task_assignment(task, chosen_host)

        # done = all(task.end_time <= self.current_time for task in self.tasks)
        # done = all(task.is_completed(self.current_time) for task in self.tasks)
        completed_tasks = sum(task.is_completed(self.current_time) for task in self.tasks)
        print(f"Curtime: {self.current_time}, Completed tasks: {completed_tasks}, Total tasks: {Task.total_tasks}")
        done = completed_tasks == Task.total_tasks
        
        # 计算新的最晚结束时间
        end_times = [task.end_time for task in self.tasks if task.end_time is not None]
        # 计算时间成本和能耗成本
        total_time_cost = max(end_times) if len(end_times)>0 else self.current_time
        total_energy_cost = self.calculate_energy_consumption()
        # 返回值
        state = self.get_state3(task, host_mask) # 状态
        reward = self.get_reward(task, chosen_host, assign_flag, total_time_cost, total_energy_cost, dw=0, done=done)  # 奖励
        info = {'assign_flag': assign_flag}  # 附加信息
        return state, reward, done, info   # state没有用的，只是为了和normal_step保持一致
    
    def task_assignment(self, task, chosen_host):
        logging.info(f"Task {task.task_name} scheduled on host {chosen_host.host_id}")
        task.start_time = self.current_time

        self.mark_task_started(task)  # 关键：记录本 tick 刚开始的任务，用于事件结算


        # 未考虑通信时间的end time = 当前时间 + 任务持续时间 / 主机速度
        task.end_time = self.current_time + task.task_duration / chosen_host.cpu_speed
        task.assigned_host = chosen_host
        chosen_host.assign_task(task)

        # # 随机分配通信时间点
        # task.communication_times = sorted(random.randint(int(task.start_time), int(task.end_time)) for _ in range(task.communicate_count))
        # print(f"Task {task.task_name} communication times list: {task.communication_times}")

        # 设置通信时间点
        task.communication_times = [task.start_time+t for t in task.communication_times]

    # 推进时间步，执行任务
    # 若任务执行完成，释放资源
    def execute_tasks(self):

        # 进行通信v1
        # 统计当前时间总的通信行为次数
        communication_behaviors = []
        # 计算每个任务的单次通信时间
        # 通信机制
        # for task in self.tasks:
        #     if task.assigned_host is not None and self.current_time in task.communication_times:
        #         # 找到通信对象
        #         communicate_tasks = []
        #         for other_task in task.job.tasks:
        #             if other_task != task and other_task.assigned_host is not None:
        #                 # 其他任务已分配，则绑定通信关系
        #                 communicate_tasks.append(other_task)
        #         print(f"Task {task.task_name} communicate with {[t.task_name for t in communicate_tasks]}")
        #         # 若不存在通信对象
        #         if not communicate_tasks:
        #             print(f"Task {task.task_name} has no communicate object")
        #             # 1、通信对象还未出现; 2、通信对象均已运行完成; 均简化为与其他主机通信
        #             communication_behaviors.append((task, task))
        #             continue
        #         # 若存在通信对象，则随机选择一个通信对象
        #         other_task = random.choice(communicate_tasks)
        #         if other_task.assigned_host != task.assigned_host:
        #             communication_behaviors.append((task, other_task))
        # if communication_behaviors:
        #     shared_bandwidth = self.bw / len(communication_behaviors)
        #     for task, other_task in communication_behaviors:
        #         communication_time = task.communicate_size / shared_bandwidth
        #         if task.end_time is not None:
        #             task.end_time += communication_time
        #             print(f"Task {task.task_name} and {other_task.task_name} communicate at {self.current_time}, communication_time: {communication_time}")
        
        # 进行通信v2
        # communicate_tasks = []
        # for task in self.tasks:
        #     if task.start_time is not None and self.current_time in task.communication_times:   # 我这个版本需要判断start_time是否为None
        #         communicate_tasks.append(task)
        # if len(communicate_tasks) > 0:
        #     shared_bandwidth = self.bw / len(communicate_tasks)  # 平均分配带宽
        #     for task in communicate_tasks:
        #         communication_time = task.communicate_size / shared_bandwidth
        #         # 按理说，start_time不为None的任务，end_time也不应该为None
        #         task.end_time += communication_time
        #         print(f"Task {task.task_name} communicate at {self.current_time}, communication_time: {communication_time}")


        # 按时间推进执行任务:推进一个时间步
        self.current_time += 1
        for host in self.hosts:
            for task in host.tasks:
                print(f"Task {task.task_name} is executing, start at {task.start_time}, end at {task.end_time}, current time: {self.current_time}")
                if task.end_time is not None and task.end_time <= self.current_time:
                    print(f"Task {task.task_name} of Job {task.job_name} executed, completed at {task.end_time}！！！")
                    host.release_resources(task)
                    logging.info(f"{self.current_time} - Task {task.task_name} of Job {task.job_name} executed, completed at {task.end_time}")
        current_energy_consumption = self.calculate_energy_consumption()
        self.total_energy_consumption += current_energy_consumption
        self.accumulatedEnergy()
        current_cost = self.calculate_energy_cost()
        self.totalEnergyPrices += current_cost
        print(f"Total energy consumption: {self.total_energy_consumption}, Current energy consumption: {current_energy_consumption}")
        

    def if_done(self):
        completed_tasks = sum(task.is_completed(self.current_time) for task in self.tasks)
        return completed_tasks == Task.total_tasks

    def get_tasks_to_schedule(self):
        # 获取当前时间需要调度的任务: 提交时间小于当前时间，且未开始执行
        return [task for task in self.tasks if task.submit_time <= self.current_time and task.start_time is None and task.exc_time is None]

    def get_tasks_to_execute(self):
        return [task for task in self.tasks if task.exc_time is not None and task.exc_time<= self.current_time and task.start_time is None]

    def get_hosts(self):
        random.shuffle(self.hosts)
        return self.hosts

    # 输出主机信息
    def print_hosts_info(self):
        for host in self.hosts:
            info = host.get_info()
            id = info['host_id']
            logging.info(f"Host {info['host_id']}: CPU={info['cpu']}, MEM={info['mem']}, GPU={info['gpu']}, Tasks={info['tasks']}")
            logging.info(f"energyConsumption:{self.energyConsumptionPerHost[id]},accumulatedEnergy:{self.accumulateEnergyConsumptionPerHost[id]},accumulatedCost:{self.accumulatedCostPerHost[id]}")

    def render(self):
        # 渲染任务进度 打印任务的开始和结束时间
        for task in self.tasks:
            print(f"Task {task.task_name}: {task.start_time} -> {task.end_time}")
            logging.info(f"Task {task.task_name}: {task.start_time} -> {task.end_time}")

    def print_statistics(self):
        # total_simulation_steps = self.current_time
        total_simulation_steps = max(task.end_time for task in self.tasks)
        total_cost = self.totalEnergyPrices  # todolist : 改为每台主机的成本*各自的运行时间
        total_task_duration = sum(task.task_duration for task in self.tasks)
        total_waiting_time = sum(task.start_time - task.submit_time for task in self.tasks)
        total_running_time = sum(task.end_time - task.start_time for task in self.tasks)

        average_task_duration = total_task_duration / len(self.tasks)
        average_waiting_time = total_waiting_time / len(self.tasks)
        average_running_time = total_running_time / len(self.tasks)

        logging.info(f"Total simulation steps: {total_simulation_steps}")
        logging.info(f"Total cost: {total_cost}")
        logging.info(f"Average task duration: {average_task_duration}")
        logging.info(f"Average waiting time: {average_waiting_time}")
        logging.info(f"Average running time: {average_running_time}")
        logging.info(f"Total energy consumption: {self.total_energy_consumption}")

        # best_F, best_step, best_avg_run_time, best_avg_wait_time, best_energy = self.get_best_result()
        # logging.info(f'best F:{best_F}')
        # logging.info(f'best_step:{best_step}')
        # logging.info(f'best_avg_run_time:{best_avg_run_time}')
        # logging.info(f'best_avg_wait_time:{best_avg_wait_time}')
        # logging.info(f'best_energy:{best_energy}')

        # 目标：最小化总能耗
        self.normalize_total_energy = self.total_energy_consumption / max_total_energy
        # 目标：最小化平均能耗
        self.normalize_avg_energy = self.total_energy_consumption / self.current_time / max_avg_energy
        # 目标：最小化总运行时间
        self.normalize_total_run_time = total_running_time / max_run_time
        # 目标：最小化平均运行时间
        self.normalize_avg_run_time = average_running_time / max_avg_run_time
        # 目标：最小化平均等待时间
        self.normalize_avg_waiting_time = average_waiting_time / max_avg_wait_time

        # 当前目标： 最小化总能耗*最小化平均运行时间
        F = self.normalize_total_energy * self.normalize_avg_run_time
        logging.info(f"e: {self.normalize_total_energy}, t: {self.normalize_avg_run_time}, F: {F}")

        if self.best_F>F:
            self.best_F = F
            self.best_step = self.current_time
            self.best_avg_run_time = average_running_time
            self.best_avg_wait_time = average_waiting_time
            self.best_energy = self.total_energy_consumption

    def get_best_result(self):
        return self.best_F, self.best_step, self.best_avg_run_time, self.best_avg_wait_time, self.best_energy

    # 用于预测资源利用率，作为状态空间中的预测信息特征
    def predict_resource_utilization(self):
        predicted_utilization = []
        current_time = self.current_time
        
        for host in self.hosts:
            # 合并运行中和待处理任务，过滤无效任务
            all_tasks = host.running_tasks + host.pending_tasks
            valid_tasks = [t for t in all_tasks if t.start_time is not None and t.end_time is not None and t.start_time <= current_time + 1 < t.end_time]
       
            # 计算基础利用率（带衰减权重）
            # 概率衰减权重：根据任务剩余时间动态调整置信度（离当前越近的任务权重越高）
            alpha = 0.2  # 衰减系数（建议0.1-0.3），对长时间任务给予更高置信度
            cpu_sum = gpu_sum = mem_sum = 0.0
            for task in valid_tasks:
                elapsed_ratio = (current_time - task.start_time) / task.task_duration
                weight = 1 - alpha * elapsed_ratio  # 越新的任务权重越高
                cpu_sum += task.plan_cpu * weight
                gpu_sum += task.plan_gpu * weight
                mem_sum += task.plan_mem * weight
            
            # 应用抢占补偿（动态调整）
            overload_threshold = 0.9
            preempt_prob = 0.3 if cpu_sum > overload_threshold else 0.0
            cpu_sum *= (1 - preempt_prob)
            
            # 资源上限约束
            cpu_pred = min(cpu_sum, 1.0)
            gpu_pred = min(gpu_sum, 1.0)
            mem_pred = min(mem_sum, 1.0)
            
            predicted_utilization.append(np.array([cpu_pred, mem_pred, gpu_pred], dtype=np.float32))
        
        return predicted_utilization
    
    # ============================= 强化学习 =============================
    def get_no_task_state(self, hosts, host_mask):
        # 获取没有container的状态
        relative_container_deployed = 0
        affinity_with_container = 0
        resource_request = [0, 0, 0]
        feature = []

        dm = self.delay_matrix
        predicted_utilization = self.predict_resource_utilization()
        for host, flag in zip(hosts, host_mask):
            if flag == 0:
                feature.extend([0]*11)  # 每个host和container组成8个特征
                continue
            cur_feat = []
            # host.idle_resource=np.array([cpu_num, mem, gpu_num]),  dm = self.scheduler.get_delay_matrix()   
            # affinity_with_container返回与目的容器的亲和性， 即返回与目的容器属于同一作业的容器个数w   dm[-1][host.index]是从主节点到target主机的延迟，即外部延迟  delay_matrix[source][target] 为 从source主机到target主机的延迟
            # host_feat = [host.idle_resource[0], host.idle_resource[1], host.idle_resource[2], host.speed[0], host.speed[1], host.speed[2], host.available_node, host.price, host.affinity_with_container(container), dm[-1][host.index]]
            cur_feat.extend([deal_with_affi(affinity_with_container), dm[-1][host.host_id], relative_container_deployed, sum(host.speed)/len(host.speed), host.available_node])
            cur_feat.extend(deal_with_resource(host.idle_resource, resource_request))    # relative_container_deployed同一个job的容器是否被部署过
            # container_feat = [container.resource_request[0], container.resource_request[1], container.resource_request[2], container.container_type, container.communicate_type, host_index, container.relative_container_deployed]
            # 添加资源利用率预测特征
            cur_feat.extend(predicted_utilization[host.host_id])
            # print("Container features: ", feat)
            cur_feat = features_normalize_func(cur_feat)
            feature.extend(cur_feat)
        # print("feature: ", feature)
        return feature

    def encode_time(self,t, period):
        sin_t = np.sin(2 * np.pi * t / period)
        cos_t = np.cos(2 * np.pi * t / period)
        return [sin_t,cos_t]

    def get_state(self, task, host_mask): # host_mask是可选主机的标志
        t = self.current_time * 10 % 1440 / 60 / 23
        feature = [t]
        dm = self.delay_matrix
        predicted_utilization = self.predict_resource_utilization()
        for host, flag in zip(self.hosts, host_mask):
            if flag == 0:
                feature.extend([0]*11)  # 每个host和container组成8个特征
                continue
            cur_feat = []
            # host.idle_resource=np.array([cpu_num, mem, gpu_num]),  dm = self.scheduler.get_delay_matrix()   
            # affinity_with_container返回与目的容器的亲和性， 即返回与目的容器属于同一作业的容器个数w   dm[-1][host.index]是从主节点到target主机的延迟，即外部延迟  delay_matrix[source][target] 为 从source主机到target主机的延迟
            # host_feat = [host.idle_resource[0], host.idle_resource[1], host.idle_resource[2], host.speed[0], host.speed[1], host.speed[2], host.available_node, host.price, host.affinity_with_container(container), dm[-1][host.index]]
            cur_feat.extend([deal_with_affi(host.affinity_with_container(task)), dm[-1][host.host_id], task.relative_container_deployed, sum(host.speed)/len(host.speed), host.available_node])
            cur_feat.extend(deal_with_resource(host.idle_resource, task.resource_request))    # relative_container_deployed同一个job的容器是否被部署过
            # 添加资源利用率预测特征
            cur_feat.extend(predicted_utilization[host.host_id])
            # container_feat = [container.resource_request[0], container.resource_request[1], container.resource_request[2], container.container_type, container.communicate_type, host_index, container.relative_container_deployed]
            # print("Container features: ", feat)   relative_container_deployed resource_request
            cur_feat = features_normalize_func(cur_feat)
            feature.extend(cur_feat)
        # print("feature: ", feature)
        return feature
    
    def get_state2(self, task, host_mask): # host_mask是可选主机的标志
        feature = []
        dm = self.delay_matrix
        for host, flag in zip(self.hosts, host_mask):
            if flag == 0:
                feature.extend([0]*8)  # 每个host和container组成8个特征
                continue
            cur_feat = []
            # host.idle_resource=np.array([cpu_num, mem, gpu_num]),  dm = self.scheduler.get_delay_matrix()   
            # affinity_with_container返回与目的容器的亲和性， 即返回与目的容器属于同一作业的容器个数w   dm[-1][host.index]是从主节点到target主机的延迟，即外部延迟  delay_matrix[source][target] 为 从source主机到target主机的延迟
            # host_feat = [host.idle_resource[0], host.idle_resource[1], host.idle_resource[2], host.speed[0], host.speed[1], host.speed[2], host.available_node, host.price, host.affinity_with_container(container), dm[-1][host.index]]
            cur_feat.extend([deal_with_affi(host.affinity_with_container(task)),task.relative_container_deployed, sum(host.speed)/len(host.speed)])
            cur_feat.extend(deal_with_resource(host.idle_resource, task.resource_request))    # relative_container_deployed同一个job的容器是否被部署过
            cur_feat.extend(host.idle_resource)   
            cur_feat.extend([host.host_id, host.gpu_type, host.price, len(host.tasks)])
            cur_feat.extend(task.resource_request)   
            cur_feat.extend([task.communicate_count, task.communicate_size, task.task_duration, task.instance_num, task.submit_time, task.gpu_type])   
            cur_feat.extend([task.job_name, task.task_name])   
            # container_feat = [container.resource_request[0], container.resource_request[1], container.resource_request[2], container.container_type, container.communicate_type, host_index, container.relative_container_deployed]
            # print("Container features: ", feat)   relative_container_deployed resource_request
            cur_feat = features_normalize_func(cur_feat)
            feature.extend(cur_feat)
        # print("feature: ", feature)
        return feature

    def get_state3(self, task, host_mask):
        t = self.current_time % 144 / 144
        feature = [t]
        for host, flag in zip(self.hosts, host_mask):
            if flag == 0:
                feature.extend([0]*3)  # 每个host和container组成8个特征
                continue
            cur_feat = host.idle_resource
            cur_feat = self.normalize3(cur_feat)
            feature.extend(cur_feat)
        return feature

    def normalize3(self, feature):
        maxM = np.array([8000,128,800])
        return np.array(feature)/maxM

    def get_state_timeDecide(self,task):
        t = self.current_time % 144 / 144
        feature = [t]
        for host in self.hosts:
            cur_feat = host.idle_resource
            cur_feat = self.normalize3(cur_feat)
            feature.extend(cur_feat)
        for que in self.timeQue:
            feature.extend(self.normalize3(que))
        taskfeature = self.normalize3(task.resource_request)
        feature.extend(taskfeature)
        f = 0.0
        f = task.decline/144
        feature.extend([f])
        return feature



    def normalize_timeDecide(self, feature):
        maxresource =  self.maxResource
        return np.array(feature)/maxresource

    def get_reward(self, task, chosen_host, assign_flag, time_cost, energy_cost, dw=0, done=0):
        hosts = self.hosts
        if self.reward_strategy == 'my_reward':  
            return self.reward_Func.my_reward(hosts, task, chosen_host, assign_flag,self)
        elif self.reward_strategy == 'time_only':
            return self.reward_Func.time_only(time_cost, assign_flag, self.tasks, done)
        elif self.reward_strategy == 'energy_only':
            return self.reward_Func.energy_only(energy_cost, assign_flag, done)
        elif self.reward_strategy == 'et_balance':
            return self.reward_Func.et_balance(time_cost, energy_cost, assign_flag, self.tasks, chosen_host, done)

    # 计算主机能耗2000
    def calculate_energy_consumption(self):
        current_total_energy = 0.0
        for host in self.hosts:
            id = host.host_id
            # 获取主机的资源利用率
            utilization = np.array([1 - host.cpu / host.max_resource[0], 1 - host.mem / host.max_resource[1], 1 - host.gpu / host.max_resource[2]], dtype=np.float32)
            # print(f"Host {host.host_id} utilization: {utilization}")
            # utilization_tensor = torch.tensor(utilization).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 计算能耗
            # with torch.no_grad():
                # 能耗最小取0
            #     energy = torch.clamp(self.energy_model(utilization_tensor), min=0).item() * (host.cpu_speed**1.5)
                # energy = torch.clamp(self.energy_model(utilization_tensor), min=0).item()
                # energy = self.energy_model(utilization_tensor).item() * host.cpu_speed
                # print(f"Host {host.host_id} energy consumption: {energy}")
            # 给定的向量
            vector = np.array([5, 2, 3], dtype=np.float32)

            # 计算点乘
            dot_product = np.dot(utilization, vector)
            if dot_product == 0:
                energy = 0
            else:
                energy = dot_product/10*host.full+host.idle
            self.energyConsumptionPerHost[id] = energy
            current_total_energy += energy

        print(f"Total energy consumption: {current_total_energy}")
        return current_total_energy

    def calculate_energy_cost(self):
        current_cost = 0.0
        for host in self.hosts:
            id = host.host_id
            cost = host.price * self.getNowPrice() * self.energyConsumptionPerHost[id]  / 6000
            self.accumulatedCostPerHost[id] += cost
            current_cost += cost
        return current_cost
        # ---- 统一口径：价格/功率/成本 ----
    def price_at(self, t: int) -> float:
        """绝对时间 t 的电价（对 144 槽取模）。"""
        return float(self.priceFunc(int(t % 144)))

    def marginal_task_power(self, task) -> float:
        """
        估算任务“边际功率”（与放置阶段口径一致或近似一致）。
        若你已有更精确的功率模型，请在这里调用以保持一致。
        """
        # 若已有专用函数，可替换为 self._estimate_task_power_share(task)
        # 这里给出一个与你现有能耗近似一致的简化：
        vec = np.array([5, 2, 3], dtype=np.float32)  # 与 calculate_energy_consumption() 同权重
        maxR = np.maximum(self.maxResource, 1.0)     # 防 0
        util = np.array(task.resource_request, dtype=np.float32) / maxR
        dot = float(np.dot(util, vec))               # 0~?
        # 取集群平均 full/idle，保证与主机能耗口径同量纲
        idle_list = [getattr(h, "idle", 18.0) for h in self.hosts]
        full_list = [getattr(h, "full", 180.0) for h in self.hosts]
        avg_idle = float(np.mean(idle_list)) if len(idle_list) else 18.0
        avg_full = float(np.mean(full_list)) if len(full_list) else 180.0
        return avg_idle + (avg_full - avg_idle) * (dot / 10.0)

    def marginal_task_cost(self, task, start_time: int, duration: int) -> float:
        """任务若从 start_time 连续运行 duration 槽的“边际成本”（边际功率×电价）。"""
        p_share = self.marginal_task_power(task)
        dur = int(duration)
        return float(sum(self.price_at(start_time + k) * p_share for k in range(dur)))

    # ---- 本 tick 刚开始任务：供时延头结算 ----
    def mark_task_started(self, task):
        """任务被放置、start_time 写入的时刻调用一次。"""
        self._started_tasks.append(task)

    def pop_started_tasks(self):
        """弹出并清空“刚开始任务”列表。"""
        tasks = self._started_tasks
        self._started_tasks = []
        return tasks

    def accumulatedEnergy(self):
        for host in self.hosts:
            id = host.host_id
            self.accumulateEnergyConsumptionPerHost[id] += self.energyConsumptionPerHost[id]

    def priceFunc(self,time):
        if time >= 0 and time < 48 :
            return 0.23
        elif 48 <= time and time < 60  :
            return 0.59
        elif 72 <= time and time < 84 :
            return 0.59
        elif 114 <= time and time < 144 :
            return 0.59
        else :
            return 1


    def getNowPrice(self):
        nowPrice =  self.priceFunc(self.current_time%144)
        return nowPrice


    def timeDecideStep(self,task, delay, action):
        ##修改时隙队列
        if delay == 1:
            self.timeQue[action] += task.resource_request
        next_state = self.get_state_timeDecide(task)
        reward = self.getRewardTimeDecide(delay)
        return next_state, reward, 0
    def getRewardTimeDecide(self,delay):
        return 0

    def toDAG(self, queueued_tasks):
        # 将当前任务列表转换为DAG表示
        dag = {}
        for task in queueued_tasks:
            if task.dag_id not in dag:
                dag[task.dag_id] = []
            dag[task.dag_id].append(task)
        return dag
    
    def isqualified(self, tasks_to_schedule):
        qualified = []
        flag = True
        for task in tasks_to_schedule:
            for parent_name in task.parent_tasks:
                parent_task = self.task_dict.get(parent_name)
                if parent_task is None:
                    logging.warning(
                        "Task %s references missing parent %s; treating it as an external completed dependency.",
                        task.task_name,
                        parent_name,
                    )
                    continue
                if not parent_task.is_completed(self.current_time):
                    flag = False
                    break
            if flag:
                qualified.append(task)
            flag = True
        return qualified
