# GAS-MARL: Green-Aware job Scheduling algorithm for HPC clusters based on Multi-Action Deep Reinforcement Learning

GAS-MARL is a green-aware job scheduling algorithm for HPC clusters based on multi-action deep reinforcement learning, which optimizes both renewable energy utilization and average bounded slowdown. This repository contains the source code of GAS-MARL and the datasets used.

## Install

All necessary packages can be installed with

```
pip install -r requirements.txt
```

## Set the configuration file

Between training and testing, we need to set the parameters of the experimental environment in the `configFile/config.ini`. Below is an example.

```
[GAS-MARL setting]
eta = 0.002
MAX_QUEUE_SIZE = 256
MAX_perNodePower=50
run_win = 64
green_win = 24
delayMaxJobNum=5
delayTimeList=[300,600,1200,1800,2400,3000,3600]

[general setting]
turbinePowerNominal = 7200
numberPv = 200
processor_per_machine = 8
idlePower = 50
```

Here’s the description for each parameter in the updated configuration:

### [training setting]

- **`eta`**: Penalty factors for model training
- **`MAX_QUEUE_SIZE`**: The maximum number of jobs in the waiting queue information.
- **`run_win`**: The maximum number of running jobs in the running jobs information.
- **`green_win`**: The number of slots in renewable energy information.
- **`delayMaxJobNum`**: The maximum number of `RN` in delay decision action type 2.
- **`delayTimeList`**: The delay time candidate list `DS` in delay decision action type 3.

### [general setting]

- **`turbinePowerNominal`**: The rated power of the wind turbine.
- **`numberPv`**: The effective irradiated area of the photovoltaic panel.
- **`processor_per_machine`**: The number of processors available per machine.
- **`idlePower`**: The power consumption (in watts) of a machine when it is idle.
- **`MAX_perNodePower`**: The maximum power consumption (in watts) allowed per processor.

## Training

Both GAS-MARL and PPO need to be trained first. Below are the command-line instructions for training the two algorithms.

The options for training GAS-MARL:

```
python MARL.py --workload [str] --backfill [int]
```

The options for training PPO:

```
python MaskablePPO.py --workload [str] --backfill [int]
```

Here are the descriptions for each option:

- **--workload:** the name of the job trace (lublin_256, Cirne, Jann)
- **--backfill:** the backfill policy, 0 = No backfilling, 1 = Greenn-backfilling, 2 = EASY-Backfilling.

## Testing

The options for testing all algorithm:

```
python compare.py --workload [str] --len [int] --iter [int] --backfill [int]
```

Here are the descriptions for each option:

- **--workload:** the name of the job trace (lublin_256, Cirne, Jann)
- **--len:** the length of the scheduling sequence
- **--iter:** the number of job sequences sampled
- **--backfill:** the backfill policy, 0 = No backfilling, 1 = Greenn-backfilling, 2 = EASY-Backfilling.

## An Example

Here is a complete example of training and testing:

First, set the `configFile/config.ini` as shown in the "Set the configuration file" section, and then start training GAS-MARL and PPO.

Train GAS-MARL with Green-Backfilling enabled:

```
python MARL.py --workload lublin_256 --backfill 1

```

Train PPO with Green-Backfilling enabled:

```
python MaskablePPO.py --workload lublin_256 --backfill 1
```

Test the performance of each algorithm with Green-Backfilling enabled:

```
python compare.py --workload lublin_256 --len 1024 --iter 10  --backfill 1 
```

## Output

The performance monitoring data of the model training process for GAS-MARL and PPO will be output separately in `MARL_[workload].csv` and `Maskable)[worklaod].csv`.

After running tests with `compare.py`, the results will be printed to the console and saved in `result.csv`.

## Citing GAS-MARL

```
If you reference or use GAS-MARL in your research, please cite:
@article{CHEN2025107760,
title = {GAS-MARL: Green-Aware job Scheduling algorithm for HPC clusters based on Multi-Action Deep Reinforcement Learning},
journal = {Future Generation Computer Systems},
volume = {167},
pages = {107760},
year = {2025},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2025.107760},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X2500055X},
author = {Rui Chen and Weiwei Lin and Huikang Huang and Xiaoying Ye and Zhiping Peng},
keywords = {Job scheduling, High-performance computing, Deep Reinforcement Learning, Renewable energy, Green computing},
}
```

### Acknowledgment

We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases:

https://github.com/DIR-LAB/deep-batch-scheduler

http://github.com/laurentphilippe/greenpower
