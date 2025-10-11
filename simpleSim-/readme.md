- ## 主函数入口 (main.py)

  主函数是项目的入口，负责解析命令行参数、初始化环境和模型，并启动调度过程。

  关键逻辑

  - 参数解析： 使用 argparse 解析命令行参数，支持选择算法（如 DQN、SAC、FirstFit 等）、奖励策略、任务排序方法、数据集路径等。
  - 数据加载： 从指定的 CSV 文件中加载任务和主机数据，创建 Task 和 Host 对象。
  - 环境初始化： 使用 SchedulingEnv 类初始化调度环境，传入任务、主机和奖励策略。
  - 模型选择： 根据 algorithm_name 参数选择模型：

  - - 启发式算法（如 FirstFit、RoundRobin）：使用 HeuristicModel。
    - 深度强化学习算法（如 DQN、SAC）：使用 DrlModel。

  - 模型训练： 调用模型的 learn 方法，执行调度和训练。

  ## 调度环境 (env.py)

  SchedulingEnv 是核心类，继承自 gym.Env，用于模拟任务调度的环境。

  关键方法

  - reset()：重置环境状态。
  - schedule_step(task, chosen_host, host_mask)：执行单个任务的调度操作。
  - normal_step()：推进时间步，更新任务和主机状态。
  - execute_tasks()：执行当前时间步的任务，释放资源。
  - get_tasks_to_schedule()：获取当前需要调度的任务。
  - get_hosts()：获取主机列表。

  ## 任务和主机管理 (task_host.py)

  定义了 Task 和 Host 类，用于描述任务和主机的属性及行为。

  关键类

  - Task：

  - - 属性：任务名称、提交时间、持续时间、资源需求等。
    - 方法：is_completed() 检查任务是否完成。

  - Host：

  - - 属性：CPU、内存、GPU 等资源信息。
    - 方法：

  - - - placement_possible(task)：检查是否可以分配任务。
      - assign_task(task)：分配任务到主机。
      - release_resources(task)：释放任务占用的资源。

  ## 模型学习 (model_learn.py)

  定义了两种模型：深度强化学习模型（DrlModel）和启发式模型（HeuristicModel）。

  关键类

  - DrlModel：

  - - learn(episode)：通过强化学习算法训练模型。
    - test()：测试模型性能，收集奖励和调度结果。

  - HeuristicModel：

  - - learn(episode)：使用启发式算法进行调度。

  - HEURISTIC_DICT 

  - - HEURISTIC_DICT 定义了多种启发式算法（如 FirstFit、RoundRobin），这些算法在 heuristic_algo.py 中实现。

  ## 启发式算法 (algorithm/heuristic_algo.py)

  实现了多种启发式调度算法。

  关键类

  - FirstFit：按主机资源从小到大排序，选择第一个满足条件的主机。
  - RoundRobin：轮询选择主机。
  - PerformenceFirst：按主机性能优先排序。
  - RandomSchedule：随机选择主机。
  - 每个类的 placement 方法实现了具体的调度逻辑。

  ## 深度强化学习算法 (algorithm/mydrl.py)

  实现了深度强化学习调度逻辑，支持多种 DRL 算法（如 DQN、SAC）。

  关键类

  - myDRL：
  - 属性：算法名称、探索率等。
  - 方法：

  - - placement(tasks_to_schedule, env, init_state)：执行调度并返回奖励和状态。
    - decay_epsilon()：探索率衰减。

  ## 奖励函数 (util/reward_func.py)

  定义了多种奖励策略，用于指导调度优化。

  关键方法

  - my_reward()：综合考虑时间和能耗的奖励。
  - time_only()：仅考虑时间成本。
  - energy_only()：仅考虑能耗成本。
  - et_balance()：平衡时间和能耗。

  ## 任务排序算法 (sort_algo.py)

  实现了多种任务排序方法。

  关键方法

  - first_submit_deploy()：按提交时间排序。
  - long_duration_first_deploy()：长任务优先。
  - short_duration_first_deploy()：短任务优先。
  - sort_tasks(algorithm_name, task_list)：根据指定算法对任务列表排序。

  ## 超参数优化 (hyper_optim.py)

  使用贝叶斯优化（gp_minimize）调整调度模型的超参数。

  关键类

  - objective_function：

  - - objective(params)：定义优化目标函数，最小化时间和能耗。

  ## 其他文件

  dataset：存放任务和主机数据集。

  log：存放日志文件。

  saved_model：存放训练好的模型。

  saved_pic：存放可视化结果。