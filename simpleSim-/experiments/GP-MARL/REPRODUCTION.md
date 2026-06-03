# GP-MARL reproduction

Paper:

Mali Imre Gergely. "Multi-Agent Deep Reinforcement Learning for Collaborative Task Scheduling." ICAART 2024.

## Paper Model

The paper models cluster scheduling as a cooperative multi-agent scheduling game:

- machines are partitioned among scheduler agents;
- all agents share pending job slots and backlog;
- each agent action selects one local machine and one shared job slot;
- PPO is trained independently for each homogeneous agent;
- local/global observations and rewards are compared;
- evaluation uses job slowdown and compares against Random, SJF, Packer, and Tetris-style heuristics.

## Local Adaptation

The current project uses `SchedulingEnv`, CSV workloads, and host objects rather than the paper's DeepRM/AEC simulator. The local adapter in `algorithm/gp_marl.py` preserves the execution structure:

- hosts are partitioned across several scheduler agents;
- each agent sees the shared ready task queue;
- each agent schedules at most one task per environment tick onto its own host partition;
- the action score combines slowdown pressure, packing efficiency, and Tetris-style resource fit.

This keeps GP-MARL comparable with the existing project baselines because it runs through the same `main.py`, `SchedulingEnv`, workload CSVs, host CSVs, reward/statistics code, and logging path.

## Example

```powershell
python main.py --algorithm_name GP-MARL --workload_path ./dataset/output.csv --host_path ./dataset/host_same.csv
```

Arrival-rate datasets can be used in the same way:

```powershell
python main.py --algorithm_name GP-MARL --workload_path ./dataset/sim_arrival_8.csv --host_path ./dataset/host_same.csv
```
