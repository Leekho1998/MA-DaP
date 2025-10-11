import random
from sim.core.scheduler import Scheduler

# 容器调度目标函数(fitness)：计算总能耗和总完成时间（Makespan）
def calculate_makespan_and_energy(schedule, tasks_cost_times, host_performance, host_energy):
    total_time = 0
    total_energy = 0
    
    for i, cost_time in enumerate(tasks_cost_times):
        host_id = schedule[1][i]  # 从第二串中获取任务的分配主机
        execution_time = cost_time / host_performance[host_id]  # 任务执行时间
        total_time = max(total_time, execution_time)  # 更新Makespan（最大执行时间）
        total_energy += host_energy[host_id] * execution_time  # 累加能耗
    
    return total_time, total_energy

# SFLA 算法：初始化种群、分组、局部搜索、全局搜索等
def sfl_algorithm(tasks, host_performance, host_energy, iterations=100, population_size=20):
    # 初始化种群（第一串：任务顺序，第二串：任务分配主机）
    population = []
    for _ in range(population_size):
        # 第一串：任务调度顺序（随机排列）
        schedule_order = random.sample(range(len(tasks)), len(tasks))
        
        # 第二串：任务分配主机（随机分配）
        host_assignment = [random.choice(range(len(host_performance))) for _ in range(len(tasks))]
        
        population.append([schedule_order, host_assignment])
    
    # 评估初始种群的适应度
    fitness = []
    for individual in population:
        makespan, energy = calculate_makespan_and_energy(individual, tasks, host_performance, host_energy)
        fitness.append((makespan, energy))
    
    # 迭代优化过程
    for iteration in range(iterations):
        # 对每个 Memeplex 进行局部搜索
        for i in range(population_size):
            # 获取当前个体
            individual = population[i]
            schedule_order, host_assignment = individual
            
            # 局部搜索：尝试改变任务分配和顺序，改进解
            new_host_assignment = host_assignment[:]
            new_host_assignment[random.randint(0, len(tasks) - 1)] = random.choice(range(len(host_performance)))
            
            new_makespan, new_energy = calculate_makespan_and_energy([schedule_order, new_host_assignment], tasks, host_performance, host_energy)
            
            # 如果新的能耗更低或调度时间更短，更新当前解
            if new_makespan < fitness[i][0] or (new_makespan == fitness[i][0] and new_energy < fitness[i][1]):
                population[i] = [schedule_order, new_host_assignment]
                fitness[i] = (new_makespan, new_energy)
        
        # 全局信息交换（模拟：随机打乱）
        random.shuffle(population)
    
    # 返回最终最优解
    best_individual = min(fitness, key=lambda x: (x[0], x[1]))  # 优先考虑Makespan，其次考虑能耗
    best_schedule = population[fitness.index(best_individual)]
    
    return best_schedule, best_individual

class SFLA(Scheduler):
    def __init__(self, sim):
        super().__init__(sim)

    def placement(self, containerids):
        self.decision = []
        self.undeployed = []
        self.deployed = []

        # 主机性能
        host_performance = []  # 20台主机，每台的性能
        # 任务耗时
        tasks_cost_times = []  # 多个任务，每个任务的耗时
        # 主机能耗
        host_energy = [] # 20台主机，每台的能耗

        best_schedule, best_fitness = sfl_algorithm(tasks_cost_times, host_performance, host_energy)

        # 打印结果
        print(f"Best schedule: {best_schedule}")  # 表示任务的最优调度方案,，包括：
        # 任务顺序 (schedule_order)：任务被执行的顺序。
        # 任务分配 (host_assignment)：每个任务被分配到的主机编号。
        print(f"Best fitness (Makespan, Energy): {best_fitness}")
        # Makespan：表示所有任务完成所需的最长时间，即任务执行的总时长。这个值越小越好，因为它表示任务被更快地完成。
        # Energy：表示完成所有任务所消耗的总能量。这个值也越小越好。



        return self.job_group_placement(containerids)
