import argparse
from model_learn import DrlModel
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np
import time
import os
import pandas as pd
import logging
from datetime import datetime
# scikit-optimize
from tqdm import tqdm

from algorithm.mydrl import myDRL
from env import SchedulingEnv
from sort_algo import sort_tasks
from task_host import Host, Job, Task

# 定义搜索空间
search_space = [Real(0, 1, name='w1'), Real(0, 1, name='w2')]

# 定义目标函数
class objective_function:
    def __init__(self, args):
        self.version_number = 0
        
        self.w1_list = []
        self.w2_list = []

        os.makedirs(args.log_path, exist_ok=True)
        # 设置日志文件路径，文件名带上当前时间戳
        log_filename = args.log_path+datetime.now().strftime('%Y%m%d_%H%M%S') + '_output.log'
        # 配置日志，设置编码为 utf-8，防止中文乱码
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        logging.getLogger().addHandler(file_handler)

        logging.info(args)
    
        # 读取数据
        workload_df = pd.read_csv(args.workload_path)
        host_df = pd.read_csv(args.host_path)

        # 创建任务和主机对象
        hosts = [Host(host_id=i, **row.to_dict()) for i, (_, row) in enumerate(host_df.iterrows())]
        job_names = workload_df['job_name'].unique()
        jobs = {job_name: Job(job_name) for job_name in job_names}
        tasks = []
        for _, row in workload_df.iterrows():
            job = jobs[row['job_name']]
            task = Task(job, **row.to_dict())
            tasks.append(task)
            job.add_task(task)

        # 设置模型
        # common config
        self.algorithm_name = args.algorithm_name
        self.reward_strategy = args.reward_name
        self.sort_name = args.sort_name
        self.episode = args.episode

        # 初始化环境
        self.env = SchedulingEnv(tasks, hosts, self.reward_strategy)

    # 定义目标函数
    def objective(self, params):  # 每次迭代都会调用这个函数
        logging.info(f"==============================version: {self.version_number}==============================")
        w1, w2 = params
        self.w1_list.append(w1)
        self.w2_list.append(w2)

        self.env.set_reward_params(w1, w2)

        model = DrlModel(self.env, self.algorithm_name, self.sort_name, iftraining=True, pref=[w1, w2])
        results_all = []

        for i in range(1, self.episode):
            logging.info(f"==============================Episode: {i}==============================")
            model.learn(episode=20) # 每次只学习一个episode，里面会自动reset env

            # test   收集测试结果
            results_r1 = []
            results_r2 = []
            collect_num = 3
            for j in tqdm(range(collect_num), desc='testing:'): # 收集10次测试结果，里面会自动reset env
                logging.info(f"==============================episode: {i} test: {j}==============================")
                # 模型测试 当前目标： 最小化总能耗*最小化平均运行时间
                normalize_total_energy, normalize_avg_run_time = model.test()
                results_r1.append(normalize_total_energy)
                results_r2.append(normalize_avg_run_time)
                # results_r2.append(self.env.average_waiting_time)
                # results_r2.append(self.env.current_time)

            results_all.append(np.multiply(results_r1, results_r2))

        F = np.mean(results_all[0])  # 优化目标,最小化F
        self.version_number += 1

        return F


def main(args):
    # 运行贝叶斯优化  n_calls表示
    result = gp_minimize(objective_function(args).objective, search_space, n_calls=10, random_state=0)  # 最多只能迭代40，不然会炸

    # 打印最佳结果
    best_params = result.x
    best_score = result.fun
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best score: {best_score}")


if __name__ == '__main__':
    """
    Schedule System.
    """
    parser = argparse.ArgumentParser(description='Schedule System')
    parser.add_argument('--algorithm_name', type=str,
                        default='DQN',
                        choices=['DQN', 'SAC', 'PPO_Discrete', 'PrefDQN'],
                        help='Name of the algorithm.')
    parser.add_argument('--reward_name', type=str,
                        default='et_balance',
                        choices=['my_reward', 'time_only', 'energy_only', 'et_balance'],
                        help='Name of the reward strategy.')
    parser.add_argument('--sort_name', type=str,
                        default='FirstSubmit',
                        choices=['FirstSubmit', 'LongFirst', 'ShortFirst'],
                        help='Method to sort tasks before placement.')
    parser.add_argument('--workload_path', type=str,
                        default='./dataset/workload_ali2025.csv',
                        help='Path to workload dataset.')
    parser.add_argument('--host_path', type=str,
                        default='./dataset/host.csv',
                        help='Path to host dataset.')
    parser.add_argument('--log_path', type=str,
                        default='./log/',
                        help='Path to save log.')
    parser.add_argument('--episode', type=int, default=2,
                        help='episode of training.')  # 每次训练一次都会测试10次
    args = parser.parse_args()

    main(args)