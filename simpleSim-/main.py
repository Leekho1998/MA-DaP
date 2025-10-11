import argparse
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import time
import os

from tqdm import tqdm

from env import SchedulingEnv
from model_learn import DrlModel, HeuristicModel, HEURISTIC_DICT
from task_host import Task, Host, Job

# 环境py38torch

def main(args):
    os.makedirs(args.log_path, exist_ok=True)
    # 设置日志文件路径，文件名带上当前时间戳
    log_filename = args.log_path+datetime.now().strftime('%Y%m%d_%H%M%S') + '_output.log'
    # 配置日志，设置编码为 utf-8，防止中文乱码
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    logging.getLogger().addHandler(file_handler)
    
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
    algorithm_name = args.algorithm_name
    reward_strategy = args.reward_name
    sort_name = args.sort_name

    # 初始化环境
    env = SchedulingEnv(tasks, hosts, reward_strategy)

    model = None
    if algorithm_name in HEURISTIC_DICT.keys():         # 启发式
        model = HeuristicModel(env, algorithm_name, sort_name)
        args.episode = 1
    else:
        model = DrlModel(env, algorithm_name, sort_name, iftraining=True)

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("start time: "+start_time)
    logging.info(args)

    model.learn(args.episode)

    print("end time: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == '__main__':
    """
    Schedule System.
    """
    parser = argparse.ArgumentParser(description='Schedule System')
    parser.add_argument('--algorithm_name', type=str,
                        default='DQN',
                        choices=['DQN', 'SAC', 'PPO_Discrete',
                                 'FirstFit', 'RoundRobin', 'PerformenceFirst', 'PerformenceLast', 'RandomSchedule'],  
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
                        choices=['./dataset/workload.csv', './dataset/workload_ali2025.csv','./dataset/workload_decline.csv','./dataset/output.csv'],
                        help='Path to workload dataset.')
    parser.add_argument('--host_path', type=str,
                        default='./dataset/host.csv',
                        help='Path to host dataset.')
    parser.add_argument('--log_path', type=str,
                        default='./log/',
                        help='Path to save log.')
    parser.add_argument('--episode', type=int, default=100,
                        help='episode of training.')
    args = parser.parse_args()

    main(args)