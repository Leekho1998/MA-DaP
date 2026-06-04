import logging
import math

import numpy as np


class REAWS:
    """
    Renewable Energy-Aware Workflow Scheduling.

    This is a simulator-level reproduction of the paper's hierarchical
    scheduler. The original method uses a global DRL agent for datacenter
    selection and local DRL agents for server selection. Here the same
    hierarchy is implemented with deterministic scoring so it can run in the
    existing DAG environment without geo-distributed MARL training.
    """

    def __init__(self, datacenter_count=4, alpha=0.45):
        self.datacenter_count = datacenter_count
        self.alpha = alpha

    def placement(self, tasks_to_schedule, env, init_state):
        task_host_pairs = {}
        hosts = env.get_hosts()
        ready_tasks = env.isqualified(tasks_to_schedule)
        datacenters = self._build_datacenters(hosts)

        for task in ready_tasks:
            dc = self._select_datacenter(task, datacenters, env)
            if dc is None:
                continue
            host = self._select_host(task, dc["hosts"], env)
            if host is None:
                continue
            task_host_pairs[task] = host
            env.task_assignment(task, host)
            logging.info(
                "REAWS assign task %s to dc %s host %s",
                task.task_name,
                dc["id"],
                host.host_id,
            )

        undeployed_tasks = [task for task in tasks_to_schedule if task not in task_host_pairs]
        info = {
            "assign_num": len(task_host_pairs),
            "task_host_pairs": task_host_pairs,
            "undeployed_tasks": undeployed_tasks,
        }
        return None, 0, False, info

    def _build_datacenters(self, hosts):
        dc_count = max(1, min(self.datacenter_count, len(hosts)))
        datacenters = [{"id": idx, "hosts": []} for idx in range(dc_count)]
        for host in hosts:
            datacenters[host.host_id % dc_count]["hosts"].append(host)
        return [dc for dc in datacenters if dc["hosts"]]

    def _select_datacenter(self, task, datacenters, env):
        scored = []
        for dc in datacenters:
            feasible_hosts = [host for host in dc["hosts"] if host.placement_possible(task)]
            if not feasible_hosts:
                continue
            green_surplus = self._green_surplus(dc, env)
            avg_speed = float(np.mean([host.cpu_speed for host in dc["hosts"]]))
            utilization = self._dc_utilization(dc)
            resource_fit = self._resource_fit(task, feasible_hosts)
            # The paper's global reward is green surplus plus local reward.
            # Since this score is minimized, green surplus is subtracted.
            score = -green_surplus - 0.20 * avg_speed + 0.65 * utilization + 0.35 * resource_fit
            scored.append((score, dc))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _select_host(self, task, hosts, env):
        feasible_hosts = [host for host in hosts if host.placement_possible(task)]
        if not feasible_hosts:
            return None

        scored = [(self._local_score(task, host, env), host) for host in feasible_hosts]
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _local_score(self, task, host, env):
        runtime = max(float(task.task_duration) / max(float(host.cpu_speed), 1e-9), 1e-9)
        transfer_time = self._predecessor_transfer_time(task, host, env)
        waiting_penalty = self._host_utilization(host) * 0.5
        execution_time = runtime + transfer_time + waiting_penalty
        energy = self._task_energy(task, host, execution_time)
        active_bonus = -0.15 if len(host.tasks) > 0 else 0.0
        balance_penalty = self._fragmentation_after(task, host)
        return self.alpha * execution_time + (1.0 - self.alpha) * energy + active_bonus + balance_penalty

    def _green_surplus(self, dc, env):
        total_power = sum(self._host_current_power(host) for host in dc["hosts"])
        green_power = self._green_power(dc["id"], env.current_time, len(dc["hosts"]))
        return green_power - total_power

    def _green_power(self, dc_id, current_time, host_count):
        # Smooth synthetic renewable supply. Different phase per synthetic DC
        # approximates location-dependent PV availability without geo data.
        day_pos = (current_time % 144) / 144.0
        phase = dc_id / max(float(self.datacenter_count), 1.0)
        solar = max(0.0, math.sin(math.pi * ((day_pos + phase) % 1.0)))
        wind = 0.35 + 0.15 * math.sin(2.0 * math.pi * (day_pos + phase * 0.5))
        renewable_ratio = min(max(0.15 + 0.65 * solar + wind, 0.0), 1.0)
        avg_full = 180.0
        return renewable_ratio * host_count * avg_full

    def _dc_utilization(self, dc):
        if not dc["hosts"]:
            return 0.0
        return float(np.mean([self._host_utilization(host) for host in dc["hosts"]]))

    def _host_utilization(self, host):
        used = host.max_resource.astype(float) - np.array([host.cpu, host.mem, host.gpu], dtype=float)
        return float(np.mean(np.clip(used / np.maximum(host.max_resource.astype(float), 1.0), 0.0, 1.0)))

    def _resource_fit(self, task, hosts):
        request = np.array([task.plan_cpu, task.plan_mem, task.plan_gpu], dtype=float)
        best_fit = 1.0
        for host in hosts:
            remaining = np.array([host.cpu, host.mem, host.gpu], dtype=float) - request
            fit = float(np.mean(np.clip(remaining / np.maximum(host.max_resource.astype(float), 1.0), 0.0, 1.0)))
            best_fit = min(best_fit, fit)
        return best_fit

    def _host_current_power(self, host):
        utilization = self._host_utilization(host)
        if utilization <= 0.0:
            return 0.0
        return float(host.idle + (host.full - host.idle) * utilization)

    def _task_energy(self, task, host, execution_time):
        request = np.array([task.plan_cpu, task.plan_mem, task.plan_gpu], dtype=float)
        max_resource = np.maximum(host.max_resource.astype(float), 1.0)
        task_utilization = float(np.mean(np.clip(request / max_resource, 0.0, 1.0)))
        active_power = host.idle + (host.full - host.idle) * task_utilization
        communication_energy = float(getattr(task, "communicate_size", 0.0)) / 1000.0 * 0.02
        return execution_time * active_power / 100.0 + communication_energy

    def _predecessor_transfer_time(self, task, host, env):
        delays = []
        for parent_name in task.parent_tasks:
            parent = env.task_dict.get(parent_name)
            if parent is None or parent.assigned_host is None:
                continue
            bandwidth = 1000.0 if parent.assigned_host.host_id == host.host_id else 600.0
            delays.append(float(getattr(task, "communicate_size", 0.0)) / bandwidth)
        return max(delays, default=0.0)

    def _fragmentation_after(self, task, host):
        remaining = np.array([host.cpu - task.plan_cpu, host.mem - task.plan_mem, host.gpu - task.plan_gpu], dtype=float)
        ratio = np.clip(remaining / np.maximum(host.max_resource.astype(float), 1.0), 0.0, 1.0)
        return float(np.std(ratio)) * 0.25
