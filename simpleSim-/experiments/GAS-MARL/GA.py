from random import shuffle
from HPCSimPickJobs import *
from PowerStruc import power_struc
import random
import os
def twoQs_function(tl):
    # print("length of TL:",len(tl))
    temp = []
    tl1 = copy.deepcopy(tl)
    tl2 = copy.deepcopy(tl)
    tl2.sort(key=lambda Task: Task.request_time, reverse=True)
    tl1.sort(key=lambda Task: Task.power * Task.request_time, reverse=True)

    while tl != []:
        i = 0
        flag1 = True
        flag2 = True
        # print(len(tl),len(tl2),len(temp))
        while flag1 and len(tl) != 0 and len(tl1) != 0:
            # print(tl[tl.index(tl1[i])])
            # print(1,len(tl),len(tl2),len(temp))
            if tl[tl.index(tl1[i])] in tl:

                temp.append(tl[tl.index(tl1[i])])
                # print(len(tl1),"from tl1")
                tl.pop(tl.index(tl1[i]))
                tl2.pop(tl2.index(tl1[i]))
                tl1.pop(i)

                flag1 = False

            else:

                tl1.pop(i)
                i += 1

        i = 0
        while flag2 and len(tl) != 0 and len(tl2) != 0:
            # print(tl[tl.index(tl2[i])])

            # print(2,len(tl),len(tl2),len(temp))
            if tl[tl.index(tl2[i])] in tl:
                # print(len(tl2),"from tl2")
                # print(i, flag2)

                temp.append(tl[tl.index(tl2[i])])
                tl.pop(tl.index(tl2[i]))
                tl1.pop(tl1.index(tl2[i]))
                tl2.pop(i)
                flag2 = False

            else:

                tl2.pop(i)
                i += 1

    return (temp)

class GA():
    def __init__(self,eta):
        self.chorm_num= 30
        self.eta=eta
        self.solutions=None
        self.fitness=None
        self.pro=None
        self.task_num=None
        self.iters_num=30

    def init(self,task_list):
        self.task_num=len(task_list)
        self.fitness=[]
        self.pro=[]
        self.solutions=[]

        tl1 = copy.deepcopy(task_list)
        sorted_list1 = sorted(tl1,key=lambda Task: Task.power * Task.request_time, reverse=True)
        self.solutions.append([tl1.index(x) for x in sorted_list1])

        tl2_before=copy.deepcopy(task_list)
        temptosendtoFunc = copy.deepcopy(task_list)
        tl2 = twoQs_function(temptosendtoFunc)
        self.solutions.append([tl2_before.index(x) for x in tl2])

        tl3 = copy.deepcopy(task_list)
        sorted_list2 = sorted(tl3,key=lambda Task: Task.request_time, reverse=True)
        self.solutions.append([tl3.index(x) for x in sorted_list2])

        tl4 = copy.deepcopy(task_list)
        sorted_list3 = sorted(tl4, key=lambda Task: Task.power, reverse=True)
        self.solutions.append([tl4.index(x) for x in sorted_list3])

        orders = list(range(len(task_list)))
        for i in range(self.chorm_num-len(self.solutions)):
            x =copy.deepcopy(orders)
            shuffle(x)
            self.solutions.append(x)

    def wheelSelection(self):
        for i in range(len(self.solutions)):
            self.pro.append(1 / float(2**(i + 1)))

    def Mutation(self,solution):
        times = random.randint(1, 5)
        y = len(solution)
        for i in range(times):
            x = random.randint(0, y - 1)
            z = random.randint(0, y - 1)
            temp = solution[x]
            solution[x] = solution[z]
            solution[z] = temp

    def Chunk_Mutation(self,solution):
        n = len(solution)

        if n < 2:
            return "error"

        block_length = random.randint(1, min(n // 2, 10))

        first_block_start = random.randint(0, n - 2 * block_length)
        second_block_start = random.randint(first_block_start + block_length, n - block_length)

        solution[first_block_start:first_block_start + block_length], solution[second_block_start:second_block_start + block_length] = \
            solution[second_block_start:second_block_start + block_length], solution[
                                                                     first_block_start:first_block_start + block_length]

        return solution

    def CrossOver(self,a,b):
        y = len(a)

        x, z = sorted(random.sample(range(y), 2))

        firstChunckNeededForOrder1 = b[:x + 1]
        firstChunckNeededForOrder2 = a[:x + 1]

        temp1 = a[x + 1:z + 1]
        temp2 = b[x + 1:z + 1]

        lastChunckNeededForOrder1 = b[z + 1:]
        lastChunckNeededForOrder2 = a[z + 1:]

        lastChunkLength = len(lastChunckNeededForOrder1)

        lastChunckNeededForOrder1.extend(firstChunckNeededForOrder1)
        lastChunckNeededForOrder1.extend(temp2)

        lastChunckNeededForOrder2.extend(firstChunckNeededForOrder2)
        lastChunckNeededForOrder2.extend(temp1)

        test1 = [rest1 for rest1 in a if rest1 not in temp1]
        test2 = [rest2 for rest2 in a if rest2 not in temp2]

        sorted_test1 = sorted(test1, key=lambda x: lastChunckNeededForOrder1.index(x))
        sorted_test2 = sorted(test2, key=lambda x: lastChunckNeededForOrder2.index(x))

        temp1.extend(sorted_test1[:lastChunkLength])
        temp2.extend(sorted_test2[:lastChunkLength])

        final1 = sorted_test1[lastChunkLength:]
        final2 = sorted_test2[lastChunkLength:]

        final1.extend(temp1)
        final2.extend(temp2)

        crossed1 = final1
        crossed2 = final2

        return crossed1, crossed2

    def run(self,env):
        temp_power = power_struc(env.cluster.statPower)
        temp_power.powerSlotLog = env.cluster.PowerStruc.getSlotFromRunning(env.running_jobs, env.current_timestamp)

        task_list=env.job_queue
        self.init(task_list)
        self.fitness = []
        for i in range(self.chorm_num):
            power_str=copy.deepcopy(temp_power)
            rwd1, greenRwd = env.getfitness(self.solutions[i],power_str)
            self.fitness.append(self.eta*rwd1+greenRwd)

        stopCounter = 0
        repeat = 0
        for iter in range(self.iters_num):
            combined = sorted(zip(self.solutions, self.fitness), key=lambda x: x[1], reverse=True)

            self.solutions = [chromosome for chromosome, _ in combined]
            self.fitness = [fitness for _, fitness in combined]
            oldBestFitness = copy.deepcopy(self.fitness[0])
            NewGeneration = []  # chromoList
            for i in range(0, 10):
                cpy = copy.deepcopy(self.solutions[i])
                NewGeneration.append(cpy)

            # wheel selection
            self.pro=[]
            self.wheelSelection()

            # Mutation

            for rep in range(15):

                w = random.random()

                for i in range(len(self.solutions)):
                    if i == 0:
                        if self.pro[i] <= w and w <= 1:
                            chosen = copy.deepcopy(self.solutions[i])
                            self.Mutation(chosen)
                            NewGeneration.append(chosen)

                    else:
                        if  self.pro[i] <= w and w <  self.pro[i-1]:
                            chosen = copy.deepcopy(self.solutions[i])
                            self.Mutation(chosen)
                            NewGeneration.append(chosen)

            # chunck Muatation
            for rep in range(15):

                w = random.random()

                for i in range(len(self.solutions)):
                    if i == 0:
                        if self.pro[i] <= w and w <= 1:
                            chosen = copy.deepcopy(self.solutions[i])
                            self.Chunk_Mutation(chosen)
                            NewGeneration.append(chosen)

                    else:
                        if self.pro[i] <= w and w <self.pro[i-1]:
                            chosen = copy.deepcopy(self.solutions[i])
                            self.Chunk_Mutation(chosen)
                            NewGeneration.append(chosen)

            # Crossover
            selected=np.zeros(len(self.solutions),dtype='int')
            for rep in range(10):

                w1 = random.random()

                for i in range(len(self.solutions)):
                    if i == 0:
                        if self.pro[i] <= w1 and w1 <= 1:
                            chosen1 = copy.deepcopy(self.solutions[i])
                            selected[i] = 1
                            indexsave1 = i

                    else:
                        if self.pro[i] <= w1 and w1 < self.pro[i-1]:
                            chosen1 = copy.deepcopy(self.solutions[i])
                            selected[i] = 1
                            indexsave1 = i

                w2 = random.random()
                for i in range(len(self.solutions)):
                    if i == 0:
                        if self.pro[i] <= w2 and w2 <= 1:
                            if selected[i] == 0:
                                chosen2 = copy.deepcopy(self.solutions[i])
                                selected[i] = 1
                                indexsave2 = i
                            else:
                                chosen2 = copy.deepcopy(self.solutions[i + 1])
                                selected[i+1] = 1
                                indexsave2 = i + 1


                    else:
                        if self.pro[i] <= w2 and w2 < self.pro[i-1]:
                            if selected[i] == 0:
                                chosen2 = copy.deepcopy(self.solutions[i])
                                selected[i] = 1
                                indexsave2 = i
                            else:
                                chosen2 = copy.deepcopy(self.solutions[i + 1])
                                selected[i+1] = 1
                                indexsave2 = i + 1

                selected[indexsave1] = 0
                selected[indexsave2] = 0


                x, y = self.CrossOver(chosen1, chosen2)

                NewGeneration.append(x)
                NewGeneration.append(y)

            # update
            self.solutions = copy.deepcopy(NewGeneration)
            self.fitness = []
            for i in range(len(self.solutions)):
                power_str = copy.deepcopy(temp_power)
                rwd1,greenRwd=env.getfitness(self.solutions[i],power_str)
                self.fitness.append(self.eta*rwd1+greenRwd)
            currentBestFitnes=max(self.fitness)
            max_fitness_chromosome = max(zip(self.solutions, self.fitness), key=lambda x: x[1])[0]
            change=currentBestFitnes-oldBestFitness
            if change == 0:
                stopCounter += 1
            else:
                stopCounter = 0
            repeat += 1

        return max_fitness_chromosome, currentBestFitnes

if __name__ == "__main__":
    seed = 0
    env = HPCEnv(backfill=1)
    ga=GA(eta=0.002)
    env.seed(seed)
    random.seed(0)
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, "./data/lublin_256.swf")
    env.my_init(workload_file=workload_file)

    traj_num=100
    o, r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0, 0, 0, 0
    running_num = 0
    t = 0
    epoch_reward = 0
    green_reward = 0
    wait_reward = 0
    beta=60

    while True:

        task_list=copy.deepcopy(env.job_queue)
        if len(task_list)==1:
            exec_seq=[0]
        else:
            ga.init(task_list)
            exec_seq,_=ga.run(env)
        for i in range(len(task_list)):
            selected_job=task_list[exec_seq[i]]
            if selected_job not in env.job_queue:
                continue
            ind=env.job_queue.index(selected_job)
            o, r, d,running_num,greenRwd = env.step_for_ga(ind, 0)

            ep_ret += r
            ep_len += 1

            green_reward += greenRwd
            wait_reward += r
            epoch_reward += r + beta * greenRwd

        if d:
            t += 1
            o, r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0, 0, 0, 0
            running_num = 0
            if t >= traj_num:
                break
    print(float(epoch_reward / traj_num))
    print(float(green_reward / traj_num))
    print(float(wait_reward / traj_num))
