
import logging
from tqdm import tqdm

from algorithm.mydrl import myDRL
from algorithm.timeDecideDRL import timeDecideDRL
from algorithm.heuristic_algo import FirstFit, RoundRobin, PerformenceFirst, RandomSchedule
from sort_algo import sort_tasks

# 启发式调度算法
HEURISTIC_DICT = { 
    'FirstFit': FirstFit(),
    'RoundRobin': RoundRobin(),
    'PerformenceFirst': PerformenceFirst(ifreverse=True),
    'RandomSchedule': RandomSchedule(),
    'PerformenceLast': PerformenceFirst(ifreverse=False)
}

class DrlModel():
    def __init__(self, env, algorithm_name, sort_name, iftraining=True, pref=None):
        self.algorithm_name = algorithm_name
        self.sort_name = sort_name
        self.iftraining = iftraining
        self.env = env
        self.algorithm = myDRL(algorithm_name, is_training=True, pref=pref)
        self.algorithm_timeDecide = timeDecideDRL('PPO_Discrete', is_training=True, pref=pref)

    def learn(self, episode):
        best_reward = 0
        for i in tqdm(range(episode)):
            logging.info("Episode: {}".format(i)) 
            # 运行调度决策
            state = self.env.reset()  # 这个信息并不会影响任何东西
            done = False

            total_reward = 0
            assign_num = 0 # 算法调度次数
            while not done:
                #把submit的任务时延
                


                # 获取要调度的任务
                tasks_to_schedule = self.env.get_tasks_to_schedule()
                # if len(tasks_to_schedule) > 0:
                #     tasks_to_schedule = sort_tasks(self.sort_name, tasks_to_schedule)
                #     self.algorithm_timeDecide.timeDecide(tasks_to_schedule, self.env, state)
                dagQueue = self.env.toDAG(tasks_to_schedule)
                ## dag={'job_name':[task1,task2,...],...}
                # tasks_to_execute = self.env.get_tasks_to_execute()
                if len(tasks_to_schedule) > 0:  # 有任务需要调度
                    tasks_to_schedule = sort_tasks(self.sort_name, tasks_to_schedule) # 对任务的排序 输入是任务列表，输出是排序后的任务列表
                    # 调度并执行任务
                    # 传入的这个state没有一点作用，拿到的state也没有用
                    #time_decide
                    state, reward, done, info = self.algorithm.placement(tasks_to_schedule, self.env, state)  # 输入是排序后的任务,
                    assign_num += info['assign_num']
                    if assign_num % 10 == 0: # 更新衰减
                        self.algorithm.decay_epsilon()
                        # self.algorithm_timeDecide.decay_epsilon()

                    # 避免除以0
                    avg_reward = reward if info['assign_num']==0 else reward/info['assign_num']
                    logging.info("time: {}, total reward: {}, assign_num: {}, avg reward: {}".format(self.env.current_time, reward, info['assign_num'], avg_reward))
                    total_reward += reward
                    if len(info['undeployed_tasks']) > 0:
                        logging.info(f"undeployed tasks len: {len(info['undeployed_tasks'])}")
                else:
                    print("No tasks to schedule at current time: {}".format(self.env.current_time))


                # 需要往前推进一个时间步

                state, reward, done, info = self.env.normal_step()    # 这4个信息并不会影响任何东西
                # 放置一步完成后
                _ = self.algorithm_timeDecide.settle_started_tasks(self.env)

                logging.info(f"done:{done}")

            avg_reward = total_reward/assign_num if assign_num!=0 else 0
            logging.info(f'assign num: {assign_num}, avg_reward of this episode {i}: {avg_reward}')
            # print("Total simulation steps: {}, Total reward: {}, Average reward: {}".format(env.current_time, sum(reward_list), avg_reward))
            if avg_reward > best_reward:  # 更新最好的reward
                self.algorithm.agent.save(model_type="best_reward_model") # 保存最好的模型

            # 渲染
            self.env.render()
            # 输出调度统计信息
            self.env.print_statistics()

        best_F, best_step, best_avg_run_time, best_avg_wait_time, best_energy = self.env.get_best_result()
        logging.info(f'best F:{best_F}')
        logging.info(f'best_step:{best_step}')
        logging.info(f'best_avg_run_time:{best_avg_run_time}')
        logging.info(f'best_avg_wait_time:{best_avg_wait_time}')
        logging.info(f'best_energy:{best_energy}')

    def test(self):
        state = self.env.reset()
        done = False
        self.algorithm.is_training = False

        total_reward = 0
        assign_num = 0 # 算法调度次数
        while not done:
            # 获取要调度的任务
            tasks_to_schedule = self.env.get_tasks_to_schedule()
            if len(tasks_to_schedule) > 0:  # 有任务需要调度
                tasks_to_schedule = sort_tasks(self.sort_name, tasks_to_schedule) # 对任务的排序 输入是任务列表，输出是排序后的任务列表
                # 调度并执行任务
                state, reward, done, info = self.algorithm.placement(tasks_to_schedule, self.env, state)  # 输入是排序后的任务,里面自带时间步推进

                assign_num += info['assign_num']
                # 避免除以0
                avg_reward = reward if info['assign_num']==0 else reward/info['assign_num']
                logging.info("time: {}, total reward: {}, assign_num: {}, avg reward: {}".format(self.env.current_time, reward, info['assign_num'], avg_reward))
                total_reward += reward
            else:
                print("No tasks to schedule at current time: {}".format(self.env.current_time))

            # 需要往前推进一个时间步
            state, reward, done, info = self.env.normal_step()  
            logging.info(f"done:{done}")

        avg_reward = total_reward/assign_num if assign_num!=0 else 0
        logging.info(f'assign num: {assign_num}, avg_reward: {avg_reward}')

        # 渲染
        self.env.render()
        # 输出调度统计信息
        self.env.print_statistics()

        return self.env.normalize_total_energy, self.env.normalize_avg_run_time

class HeuristicModel():
    def __init__(self, env, algorithm_name, sort_name):
        self.algorithm_name = algorithm_name
        self.sort_name = sort_name
        self.env = env
        
        self.algorithm = HEURISTIC_DICT[algorithm_name]

    def learn(self, episode):  # 启发式算法不需要episode参数
        state = self.env.reset()
        done = False

        assign_num = 0 # 算法调度次数
        while not done:
            # 获取要调度的任务
            tasks_to_schedule = self.env.get_tasks_to_schedule()
            if len(tasks_to_schedule) > 0:  # 有任务需要调度
                tasks_to_schedule = sort_tasks(self.sort_name, tasks_to_schedule) # 对任务的排序 输入是任务列表，输出是排序后的任务列表
                # 调度并执行任务 
                state, reward, done, info = self.algorithm.placement(tasks_to_schedule, self.env, state)  # 输入是排序后的任务 里面没有时间步推进
                assign_num += info['assign_num']
            else:
                print("No tasks to schedule at current time: {}".format(self.env.current_time))

            # 需要往前推进一个时间步
            state, reward, done, info = self.env.normal_step()  
            logging.info(f"done:{done}")

        logging.info(f'assign num: {assign_num}')

        # 渲染
        self.env.render()
        # 输出调度统计信息
        self.env.print_statistics()