# GAS-MARL reproduction

This directory vendors the official GAS-MARL implementation and datasets from:

https://github.com/sioncr/green-rl-sched

Paper:

Rui Chen, Weiwei Lin, Huikang Huang, Xiaoying Ye, Zhiping Peng. "GAS-MARL: Green-Aware job Scheduling algorithm for HPC clusters based on Multi-Action Deep Reinforcement Learning." Future Generation Computer Systems 167 (2025) 107760.

## Scope

The current project models heterogeneous task scheduling, while GAS-MARL models independent HPC batch jobs in SWF traces. To preserve the paper's assumptions, this branch keeps the official simulator as an experiment module instead of forcing it into the existing `env.py` API.

The module includes:

- `MARL.py`: GAS-MARL training with job-selection and delay actions.
- `MaskablePPO.py`: one-action PPO baseline.
- `compare.py`: paper-style comparison after trained models exist.
- `run_heuristics.py`: local smoke-test runner for FCFS, F2, LPTPN, and optional GA without requiring trained RL checkpoints.
- `data/`: Lublin-256, Cirne, Jann SWF traces plus solar/wind data.
- `configFile/config.ini`: paper configuration values.

## Install

From this directory:

```powershell
python -m pip install -r requirements.txt
```

## Smoke test

Run a small heuristic-only experiment first:

```powershell
python run_heuristics.py --workload lublin_256 --len 64 --iter 1 --backfill 1
```

Add GA for a slower baseline:

```powershell
python run_heuristics.py --workload lublin_256 --len 64 --iter 1 --backfill 1 --include-ga
```

## Paper-style training and testing

Train GAS-MARL with Green-Backfilling:

```powershell
python MARL.py --workload lublin_256 --backfill 1
```

Train PPO with Green-Backfilling:

```powershell
python MaskablePPO.py --workload lublin_256 --backfill 1
```

Evaluate all algorithms after checkpoints are available:

```powershell
python compare.py --workload lublin_256 --len 1024 --iter 10 --backfill 1
```

Repeat for `Cirne` and `Jann` to match the paper tables.
