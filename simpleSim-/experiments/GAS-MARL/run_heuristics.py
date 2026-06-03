import argparse
import os
import random

import pandas as pd

from GA import GA
from HPCSimPickJobs import HPCEnv, eta


def column_averages(rows):
    return [sum(col) / len(col) for col in zip(*rows)]


def ga_policy(env):
    ga = GA(eta=eta)
    slowdown = 0
    green_util = 0

    while True:
        task_list = list(env.job_queue)
        if len(task_list) == 1:
            exec_seq = [0]
        else:
            ga.init(task_list)
            exec_seq, _ = ga.run(env)

        done = False
        for idx in exec_seq:
            selected_job = task_list[idx]
            if selected_job not in env.job_queue:
                continue
            queue_idx = env.job_queue.index(selected_job)
            _, reward, done, _, green_reward = env.step_for_ga(queue_idx, 0)
            slowdown += reward
            green_util += green_reward

        if done:
            break

    return slowdown, green_util


def run_policy(env, sequence_len, iterations, include_ga):
    start_points = [6567, 7146, 919, 4498, 8632, 8217, 6890, 5225, 8064, 6122]
    random.seed(0)

    results = {
        "FCFS": [],
        "F2": [],
        "LPTPN": [],
    }
    if include_ga:
        results["GA"] = []

    for iter_num in range(iterations):
        start = start_points[iter_num % len(start_points)]

        env.reset_for_test(sequence_len, start)
        log, green_util = env.schedule_curr_sequence_reset(env.fcfs_score)
        slowdown = sum(log.values())
        results["FCFS"].append([slowdown, green_util, eta * slowdown + green_util])

        env.reset_for_test(sequence_len, start)
        log, green_util = env.schedule_curr_sequence_reset(env.f2_score)
        slowdown = sum(log.values())
        results["F2"].append([slowdown, green_util, eta * slowdown + green_util])

        env.reset_for_test(sequence_len, start)
        log, green_util = env.schedule_LPTPN_sequence_reset()
        slowdown = sum(log.values())
        results["LPTPN"].append([slowdown, green_util, eta * slowdown + green_util])

        if include_ga:
            env.reset_for_test(sequence_len, start)
            slowdown, green_util = ga_policy(env)
            results["GA"].append([slowdown, green_util, eta * slowdown + green_util])

    rows = []
    for algorithm, values in results.items():
        slowdown, green_util, objective = column_averages(values)
        rows.append(
            {
                "algorithm": algorithm,
                "average_bounded_slowdown": slowdown,
                "renewable_energy_utilization": green_util,
                "objective_eta_slowdown_plus_green": objective,
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="lublin_256", choices=["lublin_256", "Cirne", "Jann"])
    parser.add_argument("--len", "-l", type=int, default=64)
    parser.add_argument("--iter", "-i", type=int, default=1)
    parser.add_argument("--backfill", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--include-ga", action="store_true")
    args = parser.parse_args()

    workload_file = os.path.join(os.getcwd(), "data", args.workload + ".swf")
    env = HPCEnv(backfill=args.backfill)
    env.my_init(workload_file=workload_file)

    df = run_policy(env, args.len, args.iter, args.include_ga)
    output_path = "result_heuristics.csv"
    df.to_csv(output_path, index=False)
    print(df)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
