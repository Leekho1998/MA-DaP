import pandas as pd
import os
import random
import torch
from HPCSimPickJobs import *


from MaskablePPO import PPO
from GA import GA
from  MARL import PPO as MARL


def column_averages(matrix):
    transposed = list(zip(*matrix))

    averages = [sum(column) / len(column) for column in transposed]

    return averages

def safe_get(lst, index, default=None):
    return lst[index] if len(lst) > index else default

def load_policy(modelName,model_path):
    # handle which epoch to load from

    inputNum_size = [MAX_QUEUE_SIZE, run_win, green_win]
    featureNum_size = [JOB_FEATURES, RUN_FEATURE, GREEN_FEATURE]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if modelName=='MARL':
        model = MARL(batch_size=256, inputNum_size=inputNum_size,
                  featureNum_size=featureNum_size, device=device)
        model.load_using_model_name(model_path)

    elif modelName == 'PPO':
        model = PPO(batch_size=256, inputNum_size=inputNum_size,
                  featureNum_size=featureNum_size, device=device)
        model.load_using_model_name(model_path)

    return model


def RL_MultiAction(model,env):
    o = env.build_observation()
    running_num = 0

    reward = 0
    green_reward=0

    while True:
        lst = []
        for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
            if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                lst.append(1)
            elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                lst.append(1)
            else:
                lst.append(0)
        mask2 = np.zeros(action2_num, dtype=int)
        if running_num < delayMaxJobNum:
            mask2[running_num + 1:delayMaxJobNum + 1] = 1

        a1,a2=model.eval_action(o,lst,mask2)

        o, r, d, r2, sjf_t, f1_t, running_num, greenRwd = env.step(a1,a2)

        reward += r
        green_reward +=greenRwd
        if d:
            break
    return reward,green_reward

def RL_OneAction(model,env):
    o = env.build_observation()
    green_reward=0
    reward = 0
    while True:
        lst = []
        for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
            if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                lst.append(0)
            elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                lst.append(0)
            else:
                lst.append(1)
        for i in range(run_win):
            lst.append(0)
        for i in range(green_win):
            lst.append(0)

        a=model.eval_action(o,lst)
        o, r, d, r2, sjf_t, f1_t,running_num,greenRwd  = env.step(a, 0)

        reward += r
        green_reward += greenRwd
        if d:
            break

    return reward,green_reward

def GA_policy(env,eta):
    ga=GA(eta=eta)
    reward = 0
    green_reward=0

    while True:
        task_list = copy.deepcopy(env.job_queue)
        if len(task_list) == 1:
            exec_seq = [0]
        else:
            ga.init(task_list)
            exec_seq, _ = ga.run(env)
        for i in range(len(task_list)):
            selected_job = task_list[exec_seq[i]]
            if selected_job not in env.job_queue:
                continue
            ind = env.job_queue.index(selected_job)
            o, r, d, running_num, greenRwd = env.step_for_ga(ind, 0)

            reward += r
            green_reward += greenRwd

        if d:
            break

    return reward, green_reward

# @profile
def run_policy(env, nums, iters):
    PPO_r = []
    MARL_r = []
    fcfs_r = []
    lptpn_r=[]
    GA_r=[]
    f2_r=[]

    PPO_path=workload_name +'/MaskablePPO/'
    MARL_path=workload_name +'/MARL/'

    seed = 0
    random.seed(seed)
    start_list = [6567, 7146, 919, 4498, 8632, 8217, 6890, 5225, 8064, 6122]  # List of randomly generated scheduling start points
    for iter_num in range(0, iters):
        start = start_list[iter_num]
        env.reset_for_test(nums, start)

        log,greenRwd=env.schedule_curr_sequence_reset(env.fcfs_score)
        reward1=sum(log.values())
        fcfs_r.append([reward1,greenRwd,eta*reward1+greenRwd])

        log,greenRwd=env.schedule_curr_sequence_reset(env.f2_score)
        reward1=sum(log.values())
        f2_r.append([reward1,greenRwd,eta*reward1+greenRwd])

        log,greenRwd=env.schedule_LPTPN_sequence_reset()
        reward1=sum(log.values())
        lptpn_r.append([reward1,greenRwd,eta*reward1+greenRwd])

        reward1, greenRwd = GA_policy(env,eta=eta)
        GA_r.append([reward1, greenRwd, eta*reward1+greenRwd])
        env.reset_for_test(nums, start)

        model=load_policy('PPO', PPO_path)
        reward1, greenRwd=RL_OneAction(model,env)
        PPO_r.append([-reward1,greenRwd,eta*reward1+greenRwd])
        env.reset_for_test(nums, start)

        model=load_policy('MARL', MARL_path)
        reward1, greenRwd=RL_MultiAction(model,env)
        MARL_r.append([-reward1,greenRwd,eta*reward1+greenRwd])
        env.reset_for_test(nums, start)

    algorithms = {
        "FCFS": column_averages(fcfs_r),
        "F2": column_averages(f2_r),
        "LPTPN": column_averages(lptpn_r),
        "GA": column_averages(GA_r),
        "PPO": column_averages(PPO_r),
        "MARL": column_averages(MARL_r),
    }

    filtered_results = {
        "algorithm": [],
        "average bounded slowdown": [],
        "renewable energy utilization": []
    }

    for algo, result in algorithms.items():
        if result:
            filtered_results["algorithm"].append(algo)
            filtered_results["average bounded slowdown"].append(safe_get(result, 0))
            filtered_results["renewable energy utilization"].append(safe_get(result, 1))

    df = pd.DataFrame(filtered_results)
    df.to_csv("result.csv", index=False)
    print(df)


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='lublin_256')
    parser.add_argument('--len', '-l', type=int, default=1024)
    parser.add_argument('--iter', '-i', type=int, default=10)
    parser.add_argument('--backfill', type=int, default=1)


    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_name=args.workload
    workload_file = os.path.join(current_dir, "./data/"+workload_name+".swf")
    # initialize the environment from scratch
    env = HPCEnv(backfill=args.backfill)
    env.my_init(workload_file=workload_file)


    start = time.time()
    run_policy(env, args.len, args.iter)
