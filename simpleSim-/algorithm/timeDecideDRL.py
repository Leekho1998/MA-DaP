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
action_dim = 144  # 一天
state_dim = 40 * 3 + 1 + 144 * 3 + 3 + 1
# 160  # state dim
batch_size = 128
buffer_size = 1000
algo_agents = {
    'DQN': DQNAgent(obser_shape=(state_dim, batch_size), action_shape=action_dim, max_size=buffer_size,
                    load_file='./saved_model/dqn_model/'),  # dqn的buffer_size>batch_size
    'PrefDQN': PrefDQNAgent(obser_shape=(state_dim, batch_size), action_shape=action_dim, max_size=buffer_size,
                            load_file='./saved_model/dqn_model/'),  # dqn的buffer_size>batch_size
    # 'DDQN': DDQNAgent(obser_shape=(state_dim, batch_size), action_shape=action_dim, max_size=buffer_size),
    'SAC': SACAgent(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size, max_size=buffer_size,
                    load_file='./saved_model/sac_model/'),  # sac的buffer_size>batch_size
    # 'TD3': TD3Agent(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size, max_size=buffer_size),
    # 'A2C': A2CAgent(state_dim=state_dim, action_dim=action_dim, batch_size=5, max_size=5, load_file='./saved_model/a2c_model/'),  # a2c是一个一个训练的
    # 'DDPG': DDPGAgent(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size, max_size=buffer_size),
    'PPO_Discrete': PPO_discrete(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size,
                                 max_size=buffer_size, load_file='./saved_model/ppo_discrete_model/'),
    # ppo的buffer_size=batch_size且训完就重置
    # 'PPO': PPOAgent(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size),
}


class _Voucher:
    """记录 t0 的时延决策，在任务真正开始（τ）时一次性结算反事实差分奖励。"""
    __slots__ = ("task_key","state","next_state","action","decide_time",
                 "baseline_slot","delay_slots","logp","val","mask","settled")
    def __init__(self, task_key, state, next_state, action, decide_time,
                 baseline_slot, delay_slots, logp, val, mask):
        self.task_key = task_key
        self.state = state
        self.next_state = next_state
        self.action = int(action)
        self.decide_time = int(decide_time)
        self.baseline_slot = int(baseline_slot)  # 基线：立刻开始
        self.delay_slots = int(delay_slots)      # a
        self.logp = 0.0 if logp is None else float(logp)
        self.val  = 0.0 if val  is None else float(val)
        self.mask = None if mask is None else mask
        self.settled = False


class timeDecideDRL():
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
        self._vouchers = {}     # { id(task): _Voucher }
        self.lambda_wait = 0.01 # 等待惩罚系数（按天内比例 a/144）

        # self.epsilon_decay = 1e-4  # 探索率线性衰减

    def timeDecide(self, tasks_to_schedule, env, init_state):  # 单个时间步的调度,len(tasks_to_schedule) > 0才会进入
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
        for i, task in enumerate(tasks_to_schedule):  # 可选的主机列表
            time_mask = env.get_time_action_mask(task, max_slots=144)

            if np.sum(time_mask) == 0:  # 没有可分配的主机
                # print(f"current time: {env.current_time}, No available host for task {task.task_name}")
                continue
            action_space = np.arange(144)[time_mask]



            state = env.get_state_timeDecide(task)  # 只有这个state有用

            action, a_logprob = self.agent.get_action(state, action_space, self.is_training,
                                                      eps=self.epsilon)  # get_action
            if action > env.remaining_decline(task):
                delay = 0
            else:
                delay = 1
            #next_state, reward, done, info = env.schedule_step(task, chosen_host, time_mask, action)  # next_state没有后续作用
                task.exc_time = env.current_time + action

            next_state,reward,done = env.timeDecideStep(task, delay, action)


            # 1) 仍然设置预约时间
# task.exc_time 已在你现有逻辑中设置

    # 2) 结算奖励
            self.agent.replay_buffer.push(state, action, reward, next_state, done, a_logprob, dw=0)
            t0 = int(env.current_time)
            key = id(task)
            # 注意：ppo 的 get_action 里若能返回 (action, logp, val)，就把 a_logprob 当作 logp；val 没有就填 0
            # self._vouchers[key] = _Voucher(
            #     task_key=key,
            #     state=state,
            #     next_state=next_state,
            #     action=int(action),
            #     decide_time=t0,
            #     baseline_slot=t0,
            #     delay_slots=int(action),
            #     logp=a_logprob,   # 若你的 PPO 返回的是 log_prob
            #     val=0.0,
            #     mask=None
            # )

    # 3) 不在这里 push 到 replay_buffer；真正开始时再推

            cur_step_assign_num += 1        
            # state = next_state
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

        return state, cur_step_reward, done, # 返回的state没有用

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        logging.info(f"epsilon: {self.epsilon}")
    def settle_started_tasks(self, env):
    #取出当前 tick 刚开始的任务，计算 r_time = baseline - actual - λ_wait*(a/144)，
    #并把样本推到时间头的 replay buffer；凑满后再 update。
    
        started = env.pop_started_tasks() if hasattr(env, "pop_started_tasks") else []
        settled = 0
        for task in started:
            key = id(task)
            v = self._vouchers.get(key, None)
            if (v is None) or v.settled:
                continue
            if task.start_time is None or task.end_time is None:
                continue

            tau = int(task.start_time)
            d   = int(task.task_duration)  # 你的任务时长字段名称若是 duration，请改成 duration
            # 统一账本：边际功率 × 电价
            actual   = float(env.marginal_task_cost(task, start_time=tau,            duration=d))
            baseline = float(env.marginal_task_cost(task, start_time=v.baseline_slot, duration=d))
            diff = baseline - actual
            wait_pen = self.lambda_wait * (float(v.delay_slots) / 144.0)
            r_time = diff - wait_pen

            # push 到对应算法的 buffer（与原 push 形参一致）
            # PPO_Discrete：按你的 ppo buffer push 签名
            try:
                self.agent.replay_buffer.push(v.state, int(v.action), float(r_time), v.next_state, False, v.logp, dw=0)
            except TypeError:
                # 如果是 DQN/SAC 等，去掉 logp 参数按 (s,a,r,ns,done) 推
                self.agent.replay_buffer.push(v.state, int(v.action), float(r_time), v.next_state, False)

            # 训练触发条件与原来一致
            if self.is_training:
                if self.algorithm_name != 'PPO_Discrete' and len(self.agent.replay_buffer) >= self.agent.batch_size:
                    self.agent.update(); self.agent.save()
                if self.algorithm_name == 'PPO_Discrete' and len(self.agent.replay_buffer) == self.agent.batch_size:
                    # 对应你的 PPO：update 后会清空/重置 buffer 计数
                    self.agent.update()
                    self.agent.save()
                    # 有的实现需要把计数归零
                    if hasattr(self.agent.replay_buffer, "count"):
                        self.agent.replay_buffer.count = 0

            v.settled = True
            try: del self._vouchers[key]
            except KeyError: pass
            settled += 1
        return settled

