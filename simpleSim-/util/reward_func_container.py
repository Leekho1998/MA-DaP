
import logging
import numpy as np


class RewardFunc:
    def __init__(self):
        pass

    def get_reward(self, hosts, all_containers, container, ifaction, placement_reward, makespan=0): 
        if ifaction==False:  # 选出不存在的动作（补的0）
            return -2
        
        reward = placement_reward*3
        reward -= makespan/70
        # 运行时间  选总remaining_run_time大的job
        total_remaining_run_time = container.remaining_run_time
        for c in all_containers:
            if c.task.job.job_id == container.task.job.job_id:  # 属于同一个job
                total_remaining_run_time += c.remaining_run_time
        # print("total_remaining_run_time:", total_remaining_run_time)
        reward += total_remaining_run_time/50  # remaining_run_time是20~80多   50更好(相比20和100)
        # 亲和度   
        affi = 0
        # 1.选最大亲和度大的（亲和度也就0、1和2吧）
        for host in hosts:
            affi = max(affi, host.affinity_with_container(container))
        # 2.选亲和度之和大的
        # for host in hosts:
        #     affi += host.affinity_with_container(containers[container_action])
        reward += affi/10
        # 主机资源适配度（选出任意主机无法满足的容器） 计算能满足该容器的主机数,主机数最大20
        satisfy_num = len([host for host in hosts if host.placement_possible(container)])
        if satisfy_num==0:
            reward -= 1
        else:
            reward += satisfy_num/20
        
        return reward