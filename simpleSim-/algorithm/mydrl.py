from contextlib import ContextDecorator
import copy
import numpy as np
import pickle
import time

from drlAgents.ppodiscreteAgent import PPO_discrete
from drlAgents.dqnAgent import DQNAgent
from drlAgents.sacAgent import SACAgent
from drlAgents.prefDQNAgent import PrefDQNAgent
# from drlAgents.a2cAgent import A2CAgent

import logging
import os

# 设置模型
# common config
action_dim = 41  # host数量+不放置
state_dim = 121 #160  # state dim
batch_size = 128
buffer_size=1000
algo_agents = {
    'DQN': DQNAgent(obser_shape=(state_dim, batch_size), action_shape=action_dim, max_size=buffer_size, load_file='./saved_model/dqn_model/'), # dqn的buffer_size>batch_size
    'PrefDQN': PrefDQNAgent(obser_shape=(state_dim, batch_size), action_shape=action_dim, max_size=buffer_size, load_file='./saved_model/dqn_model/'), # dqn的buffer_size>batch_size
    # 'DDQN': DDQNAgent(obser_shape=(state_dim, batch_size), action_shape=action_dim, max_size=buffer_size),
    'SAC': SACAgent(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size, max_size=buffer_size, load_file='./saved_model/sac_model/'), # sac的buffer_size>batch_size
    # 'TD3': TD3Agent(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size, max_size=buffer_size),
    # 'A2C': A2CAgent(state_dim=state_dim, action_dim=action_dim, batch_size=5, max_size=5, load_file='./saved_model/a2c_model/'),  # a2c是一个一个训练的
    # 'DDPG': DDPGAgent(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size, max_size=buffer_size),
    'PPO_Discrete': PPO_discrete(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size, max_size=buffer_size, load_file='./saved_model/ppo_discrete_model/'),# ppo的buffer_size=batch_size且训完就重置
    # 'PPO': PPOAgent(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size),
}

class myDRL():
    def __init__(self, algorithm_name, is_training=True, pref=None):  # 虽然师兄的是5
        self.algorithm_name = algorithm_name
        self.agent = algo_agents[algorithm_name]
        if algorithm_name == 'PrefDQN':
            self.agent.set_pref(pref)

        self.is_training = is_training
        
        if not is_training:
            self.epsilon = 0.1
        else:
            self.epsilon = 1  # 初始探索率
            
        self.epsilon_min = 0.1  # 最小探索率
        self.epsilon_decay = 0.95  # 探索率衰减
        # self.epsilon_decay = 1e-4  # 探索率线性衰减
        
    def placement(self, tasks_to_schedule, env, init_state): # 单个时间步的调度,len(tasks_to_schedule) > 0才会进入
        hosts = env.get_hosts()
        # hosts按cpu从大到小排序
        # hosts.sort(key=lambda host: host.cpu, reverse=True)
        # hosts按速度从大到小排序
        hosts.sort(key=lambda host: host.speed.sum(), reverse=True)

        task_host_pairs = {}
        state = init_state  # 没有一点作用
        cur_step_reward = 0
        cur_step_assign_num = 0
        done = False
        for i, task in enumerate(tasks_to_schedule): # 可选的主机列表
            host_mask = np.zeros(len(hosts), dtype=bool)
            for i, host in enumerate(hosts):
                if host.placement_possible(task):
                    host_mask[i] = 1

            if np.sum(host_mask) == 0:  # 没有可分配的主机
                # print(f"current time: {env.current_time}, No available host for task {task.task_name}")
                continue
            action_space = np.arange(len(hosts))[host_mask]
            state = env.get_state3(task, host_mask)  # 只有这个state有用

            action, a_logprob = self.agent.get_action(state, action_space, self.is_training, eps=self.epsilon)  # get_action
            if action == 40:
                chosen_host = {}
            else :
                chosen_host = hosts[action]
            next_state, reward, done, info = env.schedule_step(task, chosen_host, host_mask,action)  # next_state没有后续作用

            if info['assign_flag']:   # 满足可分配的条件
                task_host_pairs[task] = chosen_host

            self.agent.replay_buffer.push(state, action, reward, next_state, done, a_logprob, dw=0) 
            # state = next_state 
            cur_step_assign_num += 1
            cur_step_reward += reward

            if self.is_training:
                if self.algorithm_name != 'PPO_Discrete' and len(self.agent.replay_buffer) >= self.agent.batch_size:
                    self.agent.update()
                    self.agent.save()
                # ppo的训练策略会清空
                if self.algorithm_name == 'PPO_Discrete' and len(self.agent.replay_buffer) == self.agent.batch_size:
                    self.agent.update()  # 
                    self.agent.save()
                    self.agent.replay_buffer.count = 0
        
        undeployed_tasks = [task for task in tasks_to_schedule if task not in task_host_pairs.keys()]
        info = {'assign_num': cur_step_assign_num, 'task_host_pairs': task_host_pairs, 'undeployed_tasks': undeployed_tasks}
        return state, cur_step_reward, done, info  # 返回的state没有用
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        logging.info(f"epsilon: {self.epsilon}")
            
    