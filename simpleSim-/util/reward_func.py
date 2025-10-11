import logging
import numpy as np
from sklearn.cluster import affinity_propagation

# 2处归一化：env那里和奖励那里
max_time_cost = 30
max_energy_cost = 40

class RewardFunc:
    def __init__(self):
        self.reset()

        self.w1 = 0.5
        self.w2 = 0.5

        self.max_avg_time = 0

    def reset(self):
        self.last_end_time = 0
        self.last_energy = 0

    def set_params(self, w1, w2):
        self.w1 = w1
        self.w2 = w2

    # def time_only(self, time_cost, assign_flag):
    #     reward = 0
    #     if assign_flag: # 成功分配  
    #         reward += 0.1
    #         if time_cost == self.last_end_time:  # 分配后并不会使原本的时间变长
    #             reward += 0.1
    #             self.last_end_time = time_cost
    #         else:
    #             reward += -0.1-time_cost/self.max_time_cost
    #     else:
    #         reward -= 2
    #     return reward
    
    # def energy_only(self, energy_cost, assign_flag):
    #     reward = 0
    #     if assign_flag: # 成功分配   
    #         reward += 0.1
    #         if energy_cost < self.last_energy:
    #             reward += 0.1
    #             self.last_energy = energy_cost
    #         else:
    #             reward += -0.1-energy_cost/self.max_energy_cost
    #     else:
    #         reward -= 2
    #     return reward

    def time_only(self, time_cost, assign_flag, tasks, done):
        reward = 0
        if assign_flag:
            reward += 0.2
        else:
            reward -= 1
        # -0.5的意义是抵消掉assign_flag加的1
        total_running_time = sum(task.end_time - task.start_time for task in tasks if task.end_time is not None)
        len_task = len([task for task in tasks if task.end_time is not None])
        reward += (self.last_end_time - time_cost)/max_time_cost*2 - total_running_time/(len_task+1)/30
        self.last_end_time = time_cost

        return reward
    
    def energy_only(self, energy_cost, assign_flag, done):
        reward = 0
        if assign_flag:
            reward += 0.2
        else:
            reward -= 1
        # self.last_energy - energy_cost是当前造成的能耗变化
        reward += (self.last_energy - energy_cost)/max_energy_cost*2
        
        self.last_energy = energy_cost
        return reward
    
    def et_balance(self, time_cost, energy_cost, assign_flag, tasks, result_host, done):
        eo = self.energy_only(energy_cost, assign_flag, done)
        to = self.time_only(time_cost, assign_flag, tasks, done)

        reward = self.w1*eo + self.w2*to
        # 考虑主机速度 
        # reward += sum(result_host.speed)/12
        #不要加这个，效果不好
        # if np.all(result_host.resource_usage_percentage<0.8):
        #     reward += 0.1  # 增加资源利用率正奖励
        # if np.all(result_host.resource_usage_percentage>0.9):
        #     reward -= 0.5  # 增加资源过载惩罚
        return reward
    
    # def time_only(self, time_cost, assign_flag, done):
    #     reward = 0
        
    #     # 基础分配奖励（成功分配是前提）
    #     if assign_flag:
    #         reward += 1.0  # 原0.5提升到1.0，强化基础信号
    #     else:
    #         # 无效分配的惩罚应考虑历史负载
    #         overload_penalty = sum([h.load_factor() for h in self.hosts])/len(self.hosts)
    #         return -3.0 * (1 + overload_penalty)  # 动态惩罚
        
    #     # 时间改进率（对比理论最优时间）
    #     baseline_time = self.calc_baseline_time(task)  # 需实现，如FirstFit的时间
    #     time_improve = (baseline_time - time_cost) / baseline_time
        
    #     # 分段激励
    #     if time_improve > 0:
    #         reward += 2.0 * np.tanh(5 * time_improve)  # 改进时非线性放大收益
    #     else:
    #         reward -= 1.0 * np.abs(time_improve)  # 性能下降时梯度惩罚
        
    #     # 时间趋势奖励（鼓励持续缩短）
    #     if done:  # episode结束时评估整体趋势
    #         trend = (self.initial_avg_time - self.current_avg_time) / self.initial_avg_time
    #         reward += 3.0 * np.clip(trend, 0, 1)  # 仅奖励正向趋势
        
    #     return reward
    
    # def energy_only(self, energy_cost, assign_flag, done):
    #     reward = 0
        
    #     if assign_flag:
    #         reward += 0.8  # 略低于时间奖励，平衡目标优先级
    #     else:
    #         return self.time_only(np.inf, False, done)  # 继承时间惩罚逻辑
        
    #     # 能效比计算（单位时间能耗）
    #     energy_per_second = energy_cost / max(time_cost, 1e-6)
        
    #     # 主机能效基准（需预计算各机型基准值）
    #     host_type = result_host.type
    #     baseline_energy = self.energy_baseline[host_type]
        
    #     # 能效改进奖励
    #     energy_ratio = baseline_energy / energy_per_second
    #     reward += 2.5 * np.log1p(energy_ratio)  # 对数形式避免极端值
        
    #     # 全局能耗约束
    #     if done and self.total_energy < self.energy_budget:
    #         reward += 4.0 * (1 - self.total_energy/self.energy_budget)
        
    #     return reward

    # def et_balance_improve(self, time_cost, energy_cost, assign_flag, done):
    #     # 先计算独立奖励
    #     to = self.time_only(time_cost, assign_flag, done)
    #     eo = self.energy_only(energy_cost, assign_flag, done)
        
    #     # 动态权重调整（需集成到贝叶斯优化框架）
    #     w1 = self.w1 * (1 + 0.2*np.tanh(self.time_slack))  # 时间紧迫性感知
    #     w2 = self.w2 * (1 + 0.3*(self.energy_budget - self.total_energy)/self.energy_budget)
        
    #     # 非线性融合（几何平均数加强关联性）
    #     balance_reward = np.sign(to*eo) * np.sqrt(np.abs(to*eo))
        
    #     # 时间-能耗联合约束项
    #     if assign_flag:
    #         energy_time_product = energy_cost * time_cost
    #         baseline_etp = self.baseline_etp_values[task.type]
    #         etp_ratio = baseline_etp / energy_time_product
    #         constraint_bonus = 1.5 * np.clip(etp_ratio - 1, -0.5, 1)
    #     else:
    #         constraint_bonus = 0
        
    #     return (w1*to + w2*eo) * 0.7 + balance_reward * 0.3 + constraint_bonus


    # def my_reward(self, hosts, task, result_host, assign_flag,env):  # 能收敛
    #     reward = 0
    #
    #     # # 时间惩罚-1
    #     # reward -= max_end_time/100
    #
    #     if not assign_flag:  # 不能做出动作，错误
    #         reward -= 1 - env.getNowPrice()
    #         return reward
    #
    #     # affi = result_host.affinity_with_task(task)
    #     # # print("affinity: ", affi)
    #     # if affi>0:  # 亲和度高  非常正确
    #     #     reward += affi
    #
    #     # affi_flag = 0
    #     #max_speed = 0
    #     #for host in hosts:
    #     #    max_speed = max(max_speed, sum(host.speed))
    #         # 但有其他的 更和container属于同一个job的 且适配container的主机
    #         # if host!=result_host and host.affinity_with_task(task)>affinity_propagation() and host.placement_possible(task):
    #         #     affi_flag = 1
    #         #     break
    #
    #     # 考虑主机速度
    #     #reward += sum(result_host.speed)/max_speed * 0.1
    #
    #     # 有其他被部署且被部署主机资源充足  动作非常错误
    #     # if affi_flag==1:
    #     #     reward -= 1
    #
    #     #if np.all(result_host.resource_usage_percentage<0.8):
    #     #    reward += 0.1  # 增加资源利用率正奖励
    #     #if np.all(result_host.resource_usage_percentage>0.9):
    #     #    reward -= 0.5  # 增加资源过载惩罚
    #     # oldCost = result_host.price*env.getNowPrice()*env.oldEnergyConsumptionPerHost[result_host.host_id]
    #     # newCost = result_host.price*env.getNowPrice()*env.energyConsumptionPerHost[result_host.host_id]
    #     reward -= env.getNowPrice()

        return reward


    def time_first(self, hosts, task, result_host, assign_flag,env):
        reward = 0
        if assign_flag:
            return -1
        else:
            max_speed = 0
            for host in hosts:
               max_speed = max(max_speed, sum(host.speed))
            reward += sum(result_host.speed) / max_speed * 0.1
            return reward
    def my_reward(self, hosts, task, result_host, assign_flag,env,signal = 1):
        if signal == 1:
            p = env.getNowPrice()
            price_diviation = 0.5
            price_bias = (p - price_diviation)/price_diviation
            if assign_flag:
                if p == 0.23:
                    return 1
                else:
                    return -10
            else:
                if p == 0.23:
                    return -1
                else:
                    return 10
        else:
            if assign_flag:
                return 1
            else:
                return -10