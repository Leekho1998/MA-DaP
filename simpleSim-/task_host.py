import numpy as np
np.random.seed(42)

class Job:
    total_jobs = 0
    def __init__(self, job_name):
        self.job_name = job_name
        self.tasks = []
        Job.total_jobs += 1

    def add_task(self, task):
        self.tasks.append(task)

    def get_tasks(self):
        return self.tasks

class Task:
    total_tasks = 0
    def __init__(self, job, job_name, submit_time, task_name, task_duration, instance_num, plan_cpu, plan_mem, plan_gpu, gpu_type, communicate_count, communicate_size,decline):
        self.job_name = job_name
        self.job = job
        self.submit_time = submit_time
        self.task_name = task_name
        self.instance_num = instance_num
        self.task_duration = task_duration
        self.plan_cpu = plan_cpu
        self.plan_mem = plan_mem
        self.plan_gpu = plan_gpu
        self.gpu_type = gpu_type
        self.communicate_count = communicate_count #  通信次数
        self.communicate_size = communicate_size     # 通信数据量
        self.decline = decline
        self.start_time = None
        self.end_time = None
        self.assigned_host = None
        self.exc_time = None
        # self.communication_tasks = []   # 与之通信的任务队列
        self.communication_times = []   # 通信时间队列
        Task.total_tasks += 1

        self.resource_request = np.array([plan_cpu, plan_mem, plan_gpu])

        # 随机分配通信时间点
        self.communication_times = sorted(np.random.choice(range(task_duration), communicate_count, replace=False))
        print(f"Task {task_name} communication times list: {self.communication_times}")

    def is_completed(self, current_time):
        return self.end_time is not None and self.end_time <= current_time
    
    def reset(self):
        self.start_time = None
        self.end_time = None
        self.assigned_host = None
    
    # 相关联的容器是否至少有一个被部署过
    @property
    def relative_container_deployed(self):
        for task in self.job.tasks:
            if task.start_time is not None and task != self:
                return 1
        return 0

class Host:
    def __init__(self, host_id, cpu, cpu_speed, mem, mem_speed, gpu, gpu_speed, gpu_type, price, idle, full):
        # cpu = cpu*100
        # mem = mem*100
        # gpu = gpu*100
        self.host_id = host_id
        self.cpu = cpu
        self.cpu_speed = cpu_speed
        self.mem = mem
        self.mem_speed = mem_speed
        self.gpu = gpu
        self.gpu_speed = gpu_speed
        self.gpu_type = gpu_type
        self.price = price
        self.tasks = []

        self.max_resource = np.array([cpu, mem, gpu])
        self.speed = np.array([cpu_speed, mem_speed, gpu_speed])
        self.available_node = 1

        self.running_tasks = []  # 元素格式: (start_time, end_time, cpu_util, mem_util, gpu_util)
        self.pending_tasks = []  # 已分配但未到start_time的任务
        self.env = None
        self.idle = idle
        self.full = full
    def reset(self):
        self.tasks = []
        self.cpu = self.max_resource[0]
        self.mem = self.max_resource[1]
        self.gpu = self.max_resource[2]

    def placement_possible(self, task):
        return self.cpu >= task.plan_cpu and self.mem >= task.plan_mem and self.gpu >= task.plan_gpu

    def assign_task(self, task):
        self.cpu -= task.plan_cpu
        self.mem -= task.plan_mem
        self.gpu -= task.plan_gpu
        self.tasks.append(task)

        if task.start_time <= self.env.current_time:
            self.running_tasks.append(task)
        else:
            self.pending_tasks.append(task)

    def update_task_queues(self):
        current_time = self.env.current_time
        for host in self.hosts:
            # 将pending_tasks中已到时间的任务转入running_tasks
            new_running = [t for t in host.pending_tasks if t.start_time <= current_time]
            host.running_tasks.extend(new_running)
            host.pending_tasks = [t for t in host.pending_tasks if t not in new_running]
            
            # 清理已完成的任务
            host.running_tasks = [t for t in host.running_tasks if t.end_time > current_time]

    def release_resources(self, task):
        self.cpu += task.plan_cpu
        self.mem += task.plan_mem
        self.gpu += task.plan_gpu
        self.tasks.remove(task)

    def completed_tasks_count(self, current_time):
        return sum(task.is_completed(current_time) for task in self.tasks)

    def get_info(self):
        task_names = [task.task_name for task in self.tasks]
        return {
            'host_id': self.host_id,
            'cpu': self.cpu,
            'mem': self.mem,
            'gpu': self.gpu,
            'tasks': task_names
        }
    
    def affinity_with_container(self, task):
        return 1 if self.gpu_type == task.gpu_type else 0
    
    @property
    def resource_usage_percentage(self):
        rate = np.array([self.cpu_usage, self.mem_usage, self.gpu_usage])
        rate[rate > 1] = 1
        return rate
    
    # cpu占用率
    @property
    def cpu_usage(self):
        return (self.max_resource[0] - self.cpu)/self.max_resource[0]
    
    # mem占用率
    @property
    def mem_usage(self):
        return (self.max_resource[1] - self.mem)/self.max_resource[1]
    
    # gpu占用率
    @property
    def gpu_usage(self):
        return (self.max_resource[2] - self.gpu)/self.max_resource[2]
    
    def affinity_with_task(self, task):
        if task == None:
            return 0
        affinity = 0
        job = task.job_name
        for target in self.tasks:
            if target.job_name == job  and target.task_name != task.task_name:
                affinity += 1
        return affinity
    
    # 空闲资源
    @property 
    def idle_resource(self): 
        return self.max_resource - np.array([self.cpu, self.mem, self.gpu])
    
    # 资源利用率
    @property
    def idle_resource_percentage(self):
        return self.idle_resource / self.max_resource

