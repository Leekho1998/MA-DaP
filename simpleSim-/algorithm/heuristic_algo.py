# scheduler.py
import random
import logging


# 选择遇到的第一个满足的主机（每次都重头开始选）
class FirstFit():
    def __init__(self):
        pass
    def placement(self, tasks_to_schedule, env, init_state):
        task_host_pairs = {}
        hosts = env.get_hosts()
        # 按容量reverse=True是大到小排序,小到大好点
        hosts.sort(key=lambda host: host.cpu, reverse=False)
        for task in tasks_to_schedule:
            for host in hosts:
                if host.placement_possible(task):
                    task_host_pairs[task] = host
                    env.task_assignment(task, host)
                    break 

        undeployed_tasks = [task for task in tasks_to_schedule if task not in task_host_pairs.keys()]
        info = {'assign_num': len(task_host_pairs), 'task_host_pairs': task_host_pairs, 'undeployed_tasks': undeployed_tasks}
        
        reward = 0
        state = None  # 返回啥都没关系
        done = False
        return state, reward, done, info
    
# 轮询（接上一次的继续选）
class RoundRobin():
    def __init__(self):
        self.cur_host_index = 0 # 这次要选择的主机索引

    def placement(self, tasks_to_schedule, env, init_state):
        task_host_pairs = {}
        hosts = env.get_hosts()
        for task in tasks_to_schedule:
            host = hosts[self.cur_host_index]
            if host.placement_possible(task):
                task_host_pairs[task] = host
                env.task_assignment(task, host)
                self.cur_host_index = (self.cur_host_index + 1) % len(hosts)
            else:
                index = (self.cur_host_index+1) % len(hosts)
                while index != self.cur_host_index:
                    host = hosts[index]
                    if host.placement_possible(task):
                        task_host_pairs[task] = host
                        env.task_assignment(task, host)
                        self.current_host_index = (index + 1) % len(hosts)
                        break
                    else:
                        index = (index + 1) % len(hosts)

        undeployed_tasks = [task for task in tasks_to_schedule if task not in task_host_pairs.keys()]
        info = {'assign_num': len(task_host_pairs), 'task_host_pairs': task_host_pairs, 'undeployed_tasks': undeployed_tasks}
        
        reward = 0
        state = None  # 返回啥都没关系
        done = False
        return state, reward, done, info
    
# 主机速度优先
class PerformenceFirst():
    def __init__(self, ifreverse=True):
        self.ifreverse = ifreverse

    def placement(self, tasks_to_schedule, env, init_state):
        task_host_pairs = {}
        hosts = env.get_hosts()
        # hosts按速度从大到小排序
        hosts.sort(key=lambda host: host.speed.sum(), reverse=self.ifreverse)
        for task in tasks_to_schedule:
            for host in hosts:
                if host.placement_possible(task):
                    task_host_pairs[task] = host
                    env.task_assignment(task, host)
                    break 

        undeployed_tasks = [task for task in tasks_to_schedule if task not in task_host_pairs.keys()]
        info = {'assign_num': len(task_host_pairs), 'task_host_pairs': task_host_pairs, 'undeployed_tasks': undeployed_tasks}
        
        reward = 0
        state = None  # 返回啥都没关系
        done = False
        return state, reward, done, info
    
# 随机选择主机
class RandomSchedule():
    def __init__(self):
        pass

    def placement(self, tasks_to_schedule, env, init_state):
        task_host_pairs = {}
        hosts = env.get_hosts()
        for task in tasks_to_schedule:
            available_hosts = [host for host in hosts if host.placement_possible(task)]
            if len(available_hosts)>0:
                chosen_host = random.choice(available_hosts)
                task_host_pairs[task] = chosen_host
                env.task_assignment(task, chosen_host)

        undeployed_tasks = [task for task in tasks_to_schedule if task not in task_host_pairs.keys()]
        info = {'assign_num': len(task_host_pairs), 'task_host_pairs': task_host_pairs, 'undeployed_tasks': undeployed_tasks}
        
        reward = 0
        state = None  # 返回啥都没关系
        done = False
        return state, reward, done, info
