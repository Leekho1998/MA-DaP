import logging
from collections import defaultdict

import numpy as np


class ECMWS:
    """
    Electricity Cost-aware Multiple Workflows Scheduling.

    This is a simulator-level reproduction of the ECMWS framework. The paper's
    CCRA policy network is approximated by a deadline-constrained electricity
    cost allocator so it can run in the existing DAG environment without adding
    a new PPO training pipeline.
    """

    def __init__(self, alpha=(0.4, 0.2, 0.4), beta=1.0, bandwidth=1000.0):
        self.alpha1, self.alpha2, self.alpha3 = alpha
        self.beta = beta
        self.bandwidth = bandwidth
        self._cache = {}

    def placement(self, tasks_to_schedule, env, init_state):
        task_host_pairs = {}
        hosts = env.get_hosts()
        ready_tasks = env.isqualified(tasks_to_schedule)
        if not ready_tasks:
            return None, 0, False, {
                "assign_num": 0,
                "task_host_pairs": task_host_pairs,
                "undeployed_tasks": list(tasks_to_schedule),
            }

        profiles = self._profiles(env)
        workflow_order = self._workflow_sequence(ready_tasks, profiles)
        ordered_tasks = self._task_sequence(ready_tasks, workflow_order, profiles)

        for task in ordered_tasks:
            host = self._ccra(task, hosts, env, profiles)
            if host is None:
                continue
            task_host_pairs[task] = host
            env.task_assignment(task, host)

        undeployed_tasks = [task for task in tasks_to_schedule if task not in task_host_pairs]
        info = {
            "assign_num": len(task_host_pairs),
            "task_host_pairs": task_host_pairs,
            "undeployed_tasks": undeployed_tasks,
        }
        return None, 0, False, info

    def _profiles(self, env):
        cache_key = id(env)
        task_count = len(env.tasks)
        if cache_key in self._cache and self._cache[cache_key]["task_count"] == task_count:
            return self._cache[cache_key]

        workflows = defaultdict(list)
        for task in env.tasks:
            workflows[self._workflow_id(task)].append(task)

        avg_speed = float(np.mean([max(host.cpu_speed, 1e-9) for host in env.hosts]))
        profiles = {
            "task_count": task_count,
            "workflows": workflows,
            "workflow": {},
            "task_rank": {},
            "sub_deadline": {},
        }

        workflow_metrics = {}
        for workflow_id, tasks in workflows.items():
            profile = self._build_workflow_profile(tasks, avg_speed)
            profiles["workflow"][workflow_id] = profile
            profiles["task_rank"].update(profile["rank_dp"])
            profiles["sub_deadline"].update(profile["sub_deadline"])
            workflow_metrics[workflow_id] = profile

        max_workload = max((p["workload"] for p in workflow_metrics.values()), default=1.0) or 1.0
        max_slack = max((p["slack"] for p in workflow_metrics.values()), default=1.0) or 1.0
        max_contention = max((p["contention"] for p in workflow_metrics.values()), default=1.0) or 1.0

        for workflow_id, profile in workflow_metrics.items():
            wl = profile["workload"] / max_workload
            st = profile["slack"] / max_slack
            ct = profile["contention"] / max_contention
            profile["workflow_rank"] = self.alpha1 * st + self.alpha2 * wl + self.alpha3 * ct

        self._cache[cache_key] = profiles
        return profiles

    def _build_workflow_profile(self, tasks, avg_speed):
        task_by_name = {task.task_name: task for task in tasks}
        successors = {task.task_name: [] for task in tasks}
        predecessors = {task.task_name: [] for task in tasks}
        for task in tasks:
            for parent_name in task.parent_tasks:
                if parent_name in task_by_name:
                    successors[parent_name].append(task.task_name)
                    predecessors[task.task_name].append(parent_name)

        work_time = {
            task.task_name: max(float(task.task_duration) / avg_speed, 1e-9)
            for task in tasks
        }
        trans_time = {
            task.task_name: max(float(getattr(task, "communicate_size", 0.0)) / self.bandwidth, 0.0)
            for task in tasks
        }

        est, eft = {}, {}
        for name in self._topological_names(tasks, predecessors, successors):
            task = task_by_name[name]
            if predecessors[name]:
                est[name] = max(eft[parent_name] for parent_name in predecessors[name])
            else:
                est[name] = float(task.submit_time)
            eft[name] = est[name] + trans_time[name] + work_time[name]

        levels = {}

        def level(name):
            if name in levels:
                return levels[name]
            if not successors[name]:
                levels[name] = 1
            else:
                levels[name] = max(level(child_name) for child_name in successors[name]) + 1
            return levels[name]

        for task in tasks:
            level(task.task_name)

        level_counts = defaultdict(int)
        for lv in levels.values():
            level_counts[lv] += 1

        rank_dp = {}

        def dp_rank(name):
            if name in rank_dp:
                return rank_dp[name]
            if not successors[name]:
                rank_dp[name] = work_time[name]
                return rank_dp[name]
            lv = levels[name]
            prev_count = max(level_counts.get(lv - 1, 1), 1)
            bottleneck = self.beta * level_counts[lv] / prev_count
            rank_dp[name] = (
                max(dp_rank(child_name) + trans_time[child_name] for child_name in successors[name])
                + work_time[name]
                + bottleneck
            )
            return rank_dp[name]

        for task in tasks:
            dp_rank(task.task_name)

        root_names = [task.task_name for task in tasks if not predecessors[task.task_name]]
        root_rank = max((rank_dp[name] for name in root_names), default=max(rank_dp.values(), default=1.0))
        root_rank = max(root_rank, 1e-9)
        workflow_deadline = max(float(task.submit_time) + float(task.decline) for task in tasks)
        workflow_submit = min(float(task.submit_time) for task in tasks)
        deadline_span = max(workflow_deadline - workflow_submit, root_rank)
        sub_deadline = {}
        for task in tasks:
            name = task.task_name
            offset = deadline_span * (root_rank - rank_dp[name] + work_time[name]) / root_rank
            sub_deadline[task] = workflow_submit + max(offset, work_time[name])

        intervals = sorted((est[task.task_name], eft[task.task_name]) for task in tasks)
        contention = 1
        for idx, (_, end_time) in enumerate(intervals):
            overlap = 1
            for later_start, _ in intervals[idx + 1:]:
                if later_start < end_time:
                    overlap += 1
            contention = max(contention, overlap)

        workload = sum(float(task.task_duration) for task in tasks)
        earliest_finish = max(eft.values(), default=workflow_submit)
        slack = max(workflow_deadline - earliest_finish, 0.0)
        return {
            "successors": successors,
            "predecessors": predecessors,
            "rank_dp": {task_by_name[name]: rank for name, rank in rank_dp.items()},
            "sub_deadline": sub_deadline,
            "workload": workload,
            "slack": slack,
            "contention": float(contention),
            "workflow_rank": 0.0,
        }

    def _topological_names(self, tasks, predecessors, successors):
        remaining = {task.task_name for task in tasks}
        ready = sorted(name for name in remaining if not predecessors[name])
        order = []
        while ready:
            name = ready.pop(0)
            if name not in remaining:
                continue
            remaining.remove(name)
            order.append(name)
            for child_name in successors[name]:
                if child_name in remaining and all(parent not in remaining for parent in predecessors[child_name]):
                    ready.append(child_name)
            ready.sort()
        order.extend(sorted(remaining))
        return order

    def _workflow_sequence(self, tasks, profiles):
        workflow_ids = {self._workflow_id(task) for task in tasks}
        return sorted(
            workflow_ids,
            key=lambda workflow_id: profiles["workflow"][workflow_id]["workflow_rank"],
        )

    def _task_sequence(self, tasks, workflow_order, profiles):
        workflow_pos = {workflow_id: idx for idx, workflow_id in enumerate(workflow_order)}
        return sorted(
            tasks,
            key=lambda task: (
                workflow_pos[self._workflow_id(task)],
                -profiles["task_rank"].get(task, 0.0),
                task.submit_time,
                task.task_name,
            ),
        )

    def _ccra(self, task, hosts, env, profiles):
        candidates = [host for host in hosts if host.placement_possible(task)]
        if not candidates:
            return None

        sub_deadline = profiles["sub_deadline"].get(task, float(task.submit_time) + float(task.decline))
        scored = [(self._allocation_score(task, host, env, sub_deadline), host) for host in candidates]
        scored.sort(key=lambda item: item[0])
        chosen = scored[0][1]
        logging.info(
            "ECMWS assign task %s to host %s with score %.6f",
            task.task_name,
            chosen.host_id,
            scored[0][0],
        )
        return chosen

    def _allocation_score(self, task, host, env, sub_deadline):
        runtime = max(float(task.task_duration) / max(float(host.cpu_speed), 1e-9), 1e-9)
        start_time = float(env.current_time)
        transfer_delay = self._predecessor_transfer_delay(task, host, env)
        finish_time = start_time + transfer_delay + runtime
        violation = max(0.0, finish_time - sub_deadline) / runtime
        electricity_cost = self._electricity_cost(task, host, env, int(np.ceil(runtime + transfer_delay)))
        remaining = np.array([host.cpu - task.plan_cpu, host.mem - task.plan_mem, host.gpu - task.plan_gpu], dtype=float)
        max_resource = np.maximum(host.max_resource.astype(float), 1.0)
        fragmentation = float(np.std(remaining / max_resource))
        affinity_penalty = 0.0 if host.affinity_with_container(task) else 0.05
        return electricity_cost * (1.0 + violation) + fragmentation + affinity_penalty

    def _electricity_cost(self, task, host, env, duration):
        duration = max(duration, 1)
        before = self._host_power(host, np.array([0.0, 0.0, 0.0]))
        after = self._host_power(host, task.resource_request.astype(float))
        marginal_power = max(after - before, after * 0.1, 1e-9)
        price_sum = sum(env.price_at(env.current_time + step) for step in range(duration))
        return float(marginal_power * price_sum * getattr(host, "price", 1.0) / 6000.0)

    def _host_power(self, host, extra_request):
        used = host.max_resource.astype(float) - np.array([host.cpu, host.mem, host.gpu], dtype=float) + extra_request
        utilization = np.clip(used / np.maximum(host.max_resource.astype(float), 1.0), 0.0, 1.0)
        if float(np.sum(utilization)) == 0.0:
            return 0.0
        return float(np.dot(utilization, np.array([5.0, 2.0, 3.0])) / 10.0 * host.full + host.idle)

    def _predecessor_transfer_delay(self, task, host, env):
        delays = []
        for parent_name in task.parent_tasks:
            parent = env.task_dict.get(parent_name)
            if parent is None or parent.assigned_host is None:
                continue
            if parent.assigned_host.host_id != host.host_id:
                delays.append(float(getattr(task, "communicate_size", 0.0)) / self.bandwidth)
        return max(delays, default=0.0)

    def _workflow_id(self, task):
        return getattr(task, "dag_id", None) or getattr(task, "job_name", None)
