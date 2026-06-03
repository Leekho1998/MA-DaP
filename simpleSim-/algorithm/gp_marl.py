import logging


class GPMARL:
    """
    GP-MARL adapter for this project's SchedulingEnv.

    The paper models task scheduling as a cooperative multi-agent game where
    each PPO agent controls a partition of machines and selects a job-slot and
    local machine action. This implementation preserves that execution pattern
    in the local simulator: hosts are split across agents, each agent observes
    the shared ready queue, then schedules at most one task onto its own hosts.
    """

    def __init__(self, num_agents=4, job_slot_size=5, objective="tetris"):
        self.num_agents = num_agents
        self.job_slot_size = job_slot_size
        self.objective = objective

    def placement(self, tasks_to_schedule, env, init_state):
        task_host_pairs = {}
        ready_tasks = [task for task in tasks_to_schedule if task.start_time is None]
        if not ready_tasks:
            return None, 0, env.if_done(), {
                "assign_num": 0,
                "task_host_pairs": task_host_pairs,
                "undeployed_tasks": [],
            }

        partitions = self._partition_hosts(env.hosts)
        shared_slots = self._job_slots(ready_tasks, env)

        for agent_id, hosts in enumerate(partitions):
            available_slots = [task for task in shared_slots if task.start_time is None]
            action = self._select_action(available_slots, hosts, env)
            if action is None:
                continue
            task, host = action
            env.task_assignment(task, host)
            task_host_pairs[task] = host
            logging.info(
                "GP-MARL agent=%s assign task=%s host=%s time=%s",
                agent_id,
                task.task_name,
                host.host_id,
                env.current_time,
            )

        undeployed_tasks = [task for task in ready_tasks if task.start_time is None]
        return None, 0, env.if_done(), {
            "assign_num": len(task_host_pairs),
            "task_host_pairs": task_host_pairs,
            "undeployed_tasks": undeployed_tasks,
        }

    def _partition_hosts(self, hosts):
        partitions = [[] for _ in range(max(1, self.num_agents))]
        ordered_hosts = sorted(hosts, key=lambda host: host.host_id)
        for idx, host in enumerate(ordered_hosts):
            partitions[idx % len(partitions)].append(host)
        return [partition for partition in partitions if partition]

    def _job_slots(self, tasks, env):
        return sorted(
            tasks,
            key=lambda task: (
                -(env.current_time - task.submit_time),
                task.task_duration,
                task.task_name,
            ),
        )[:self.job_slot_size]

    def _select_action(self, tasks, hosts, env):
        best = None
        best_score = None
        for host in hosts:
            if not hosts:
                continue
            for task in tasks:
                if task.start_time is not None or not host.placement_possible(task):
                    continue
                score = self._score(task, host, env)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (task, host)
        return best

    def _score(self, task, host, env):
        wait = max(0, env.current_time - task.submit_time)
        slowdown_pressure = (wait + task.task_duration) / max(task.task_duration, 1)
        speed = max(float(sum(host.speed)), 1.0)
        fit = self._resource_fit(task, host)
        pack = self._packing_score(task, host)

        if self.objective == "sjf":
            return -task.task_duration + 0.2 * slowdown_pressure
        if self.objective == "packer":
            return pack + 0.1 * slowdown_pressure
        return 0.45 * fit + 0.35 * pack + 0.20 * slowdown_pressure + 0.05 * speed

    def _resource_fit(self, task, host):
        cpu_fit = task.plan_cpu / max(host.cpu, 1)
        mem_fit = task.plan_mem / max(host.mem, 1)
        gpu_fit = task.plan_gpu / max(host.gpu, 1)
        return min(cpu_fit, 1.0) + min(mem_fit, 1.0) + min(gpu_fit, 1.0)

    def _packing_score(self, task, host):
        after_cpu = max(host.cpu - task.plan_cpu, 0) / max(host.max_resource[0], 1)
        after_mem = max(host.mem - task.plan_mem, 0) / max(host.max_resource[1], 1)
        after_gpu = max(host.gpu - task.plan_gpu, 0) / max(host.max_resource[2], 1)
        # Lower leftover fragmentation is better.
        return -(after_cpu + after_mem + after_gpu)
