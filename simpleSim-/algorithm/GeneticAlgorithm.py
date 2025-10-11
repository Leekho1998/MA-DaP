import random
import statistics
from sim.core.scheduler import Scheduler

class GeneticAlgorithmScheduler(Scheduler):
    def __init__(self, sim):
        super().__init__(sim)
        self.population_size = 100  # 种群大小
        self.mutation_rate = 0.01  # 变异率
        self.crossover_rate = 0.7  # 交叉率
        self.generations = 100  # 迭代次数
        
    def initialize_population(self):
        # 初始化种群
        population = []
        for _ in range(self.population_size):
            placement = {cid: random.choice(self.hosts).id for cid in self.containerids}
            population.append(placement)
        return population

    def fitness(self, placement):
        # 计算适应度，优先考虑填满单个主机
        host_loads = [0]*len(self.hosts)
        for cid, host_id in placement.items():
            host_loads[host_id] += 1
        
        # 找到最大负载的主机，鼓励填满单个主机
        max_load = max(host_loads) if host_loads else 0
        # 使用最大负载作为适应度的主要指标，最大负载越高，适应度越高
        # 同时，为了防止完全忽视其他主机，可以通过平均负载进行微调，以避免极端情况
        avg_load = statistics.mean(host_loads) if host_loads else 0
        # 调整权重，根据实际需求调整alpha和beta的值
        alpha = 1.0  # 填满主机的权重
        beta = 0.1   # 平衡其他主机负载的权重
        return alpha * max_load + beta * avg_load

    def selection(self):
        # 实现选择算法，这里简化为直接运行遗传算法流程并返回结果
        population = self.initialize_population()
        for _ in range(self.generations):
            population = self.evolve(population)
        
        # 从最优解中分离已部署和未部署的容器（此处简化处理，实际上应基于某种决策）
        best_solution = max(population, key=self.fitness)
        deployed = [cid for cid, host_id in best_solution.items() if host_id is not None]
        undeployed = [cid for cid in self.containerids if cid not in deployed]
        return deployed, undeployed

    def evolve(self, population):
        # 简化的进化过程，包括选择、交叉、变异
        new_population = []
        for _ in range(len(population)):
            parent1, parent2 = random.sample(population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

    def crossover(self, parent1, parent2):
        # 交叉操作
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(self.containerids) - 1)
            child = {**parent1, **{cid: parent2[cid] for cid in parent2 if cid not in parent1}}
            return child
        else:
            return parent1  # 或者parent2，这里简化处理

    def mutate(self, individual):
        # 变异操作
        for cid, host_id in individual.items():
            if random.random() < self.mutation_rate:
                new_host_id = random.choice([h.id for h in self.hosts if h.id != host_id])
                individual[cid] = new_host_id
        return individual

    def placement(self, containerids):
        # 假设此方法用于执行调度决策，此处简化处理，直接返回决策结果
        # 实际应用中可能需要与模拟环境交互来真正部署容器
        return containerids  # 已经在selection中做出决策，此处直接返回

    # def selection(self):   # 选择算法
    #     select = self.get_undeployed_containers()
    #     return [], [container.id for container in select]

    # # 选择操作：轮盘赌选择
    # def selection__(self):
    #     # 计算每个个体的适应度
    #     fitnesses = [self.fitness(individual) for individual in self.population]
    #     total_fitness = sum(fitnesses)
    #     # 计算每个个体的选择概率
    #     selection_probs = [fitness / total_fitness for fitness in fitnesses]
    #     # 轮盘赌选择
    #     selected = random.choices(self.population, weights=selection_probs, k=self.population_size)
    #     return selected

    # def selection(self, population):
    #     # 基于适应度的选择，例如：轮盘赌选择
    #     probabilities = [fitness / sum(fitnesses) for fitness in [self.fitness(p) for p in population]]
    #     selected = [population[i] for i in random.choices(range(len(population)), probabilities)]
    #     return selected

    # # 部署操作：基于遗传算法的部署
    # def placement(self, containerids):
    #     # 初始化种群
    #     self.population = self.init_population(containerids)
    #     # 进行多代迭代
    #     for _ in range(self.generations):
    #         # 选择
    #         selected = self.selection()
    #         # 交叉
    #         offspring = self.crossover(selected)
    #         # 变异
    #         mutated = self.mutation(offspring)
    #         # 更新种群
    #         self.population = mutated
    #     # 返回最优解
    #     best_individual = max(self.population, key=self.fitness)
    #     return self.decode(best_individual)

    # def init_population(self):
    #     # This method should return a list of individuals (i.e., solutions), each of which is a possible schedule.
    #     # The specifics of this would depend on how you've chosen to represent schedules.
    #     # Here's a simple example where each schedule is a dict mapping task IDs to resource IDs.
    #     tasks = self.sim.get_tasks()
    #     resources = self.sim.get_resources()
    #     return [self.init_individual(tasks, resources) for _ in range(self.population_size)]

    # def init_individual(self, tasks, resources):
    #     # This method should return a single individual (i.e., solution), which is a possible schedule.
    #     # The specifics of this would depend on how you've chosen to represent schedules.
    #     # Here's a simple example where each schedule is a dict mapping task IDs to resource IDs.
    #     return {task: random.choice(resources) for task in tasks}

    # def fitness(self, individual):
    #     # This method should return a fitness value for the individual (i.e., solution).
    #     # The specifics of this would depend on the specifics of your problem.
    #     # Here's a simple example where the fitness is the negative total cost of the schedule.
    #     return -self.sim.total_cost(individual)

    # def mutate(self, individual):
    #     # In genetic algorithms, mutation involves randomly altering some part of the individual.
    #     # Here we randomly reassign one of the tasks in the schedule to a different resource.
    #     if random.random() < self.mutation_rate:
    #         task = random.choice(list(individual.keys()))
    #         resource = random.choice(self.sim.get_resources())
    #         individual[task] = resource
    #     return individual