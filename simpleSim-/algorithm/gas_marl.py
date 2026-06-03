import logging


class GASMARL:
    """
    GAS-MARL adapter for this project's SchedulingEnv.

    The paper's original implementation targets SWF HPC traces. This adapter keeps
    the same scheduling idea in the local task/host environment: select a task,
    decide whether to delay it when brown-energy cost is high, and use green-style
    backfilling for low-cost tasks that can run now.
    """

    def __init__(self, delay_price_threshold=1.0, max_delay=72, backfill_limit=8):
        self.delay_price_threshold = delay_price_threshold
        self.max_delay = max_delay
        self.backfill_limit = backfill_limit

    def placement(self, tasks_to_schedule, env, init_state):
        task_host_pairs = {}
        undeployed_tasks = []

        ready_tasks = [task for task in tasks_to_schedule if task.start_time is None]
        executable = [task for task in ready_tasks if self._best_host(task, env) is not None]
        if not executable:
            return None, 0, False, {
                "assign_num": 0,
                "task_host_pairs": task_host_pairs,
                "undeployed_tasks": ready_tasks,
            }

        selected = self._select_task(executable, env)
        if self._should_delay(selected, env):
            logging.info(
                "GAS-MARL delay task=%s at time=%s price=%.2f",
                selected.task_name,
                env.current_time,
                env.getNowPrice(),
            )
            undeployed_tasks.append(selected)
        else:
            self._assign(selected, env, task_host_pairs)

        # Green-Backfilling: fill the delay/resource gap with low resource-time-power tasks.
        backfill_candidates = [
            task for task in executable
            if task is not selected and task.start_time is None and self._accept_backfill(task, env)
        ]
        backfill_candidates.sort(key=lambda task: self._backfill_score(task, env))
        for task in backfill_candidates[:self.backfill_limit]:
            if self._best_host(task, env) is None:
                undeployed_tasks.append(task)
                continue
            self._assign(task, env, task_host_pairs)

        for task in ready_tasks:
            if task.start_time is None and task not in undeployed_tasks:
                undeployed_tasks.append(task)

        done = env.if_done()
        reward = 0
        return None, reward, done, {
            "assign_num": len(task_host_pairs),
            "task_host_pairs": task_host_pairs,
            "undeployed_tasks": undeployed_tasks,
        }

    def _assign(self, task, env, task_host_pairs):
        if task.start_time is not None:
            return False
        host = self._best_host(task, env)
        if host is None:
            return False
        task_host_pairs[task] = host
        env.task_assignment(task, host)
        logging.info(
            "GAS-MARL assign task=%s host=%s time=%s price=%.2f",
            task.task_name,
            host.host_id,
            env.current_time,
            env.getNowPrice(),
        )
        return True

    def _best_host(self, task, env):
        candidates = [host for host in env.hosts if host.placement_possible(task)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda host: (
                self._host_incremental_energy(host, task),
                host.price,
                host.host_id,
            ),
        )

    def _select_task(self, tasks, env):
        return min(tasks, key=lambda task: self._selection_score(task, env))

    def _selection_score(self, task, env):
        wait = max(0, env.current_time - task.submit_time)
        urgency = wait / max(float(getattr(task, "decline", 1) or 1), 1.0)
        return (
            self._brown_cost(task, env)
            + 0.02 * task.task_duration
            - 5.0 * urgency
        )

    def _backfill_score(self, task, env):
        resource_product = max(task.plan_cpu, 1) * max(task.plan_mem, 1) * max(task.plan_gpu, 1)
        return task.task_duration * resource_product * self._brown_cost(task, env)

    def _should_delay(self, task, env):
        wait = max(0, env.current_time - task.submit_time)
        deadline_slack = max(0, getattr(task, "decline", self.max_delay) - wait)
        high_price = env.getNowPrice() >= self.delay_price_threshold
        has_slack = deadline_slack > 0 and wait < self.max_delay
        return high_price and has_slack

    def _brown_cost(self, task, env):
        duration = max(1, int(task.task_duration))
        return env.marginal_task_cost(task, int(env.current_time), duration)

    def _accept_backfill(self, task, env):
        # Low-price slots are treated as renewable-friendly in this environment.
        if env.getNowPrice() < self.delay_price_threshold:
            return True
        duration = max(1, int(task.task_duration))
        green_reference = env.marginal_task_power(task) * duration * 0.59
        return self._brown_cost(task, env) <= green_reference

    def _host_incremental_energy(self, host, task):
        cpu = task.plan_cpu / max(host.max_resource[0], 1)
        mem = task.plan_mem / max(host.max_resource[1], 1)
        gpu = task.plan_gpu / max(host.max_resource[2], 1)
        return 5 * cpu + 2 * mem + 3 * gpu
