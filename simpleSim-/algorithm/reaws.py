import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class _ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.shared(state)
        return self.actor(x), self.critic(x)


class _MaskedActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def select_action(self, state, valid_actions, is_training=True):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(state_t)
        mask = torch.full_like(logits, -1e9)
        mask[0, list(valid_actions)] = 0.0
        probs = torch.softmax(logits + mask, dim=-1)

        if is_training:
            dist = Categorical(probs=probs)
            action_t = dist.sample()
        else:
            action_t = torch.argmax(probs, dim=-1)
            dist = Categorical(probs=probs)

        log_prob = dist.log_prob(action_t)
        entropy = dist.entropy()
        return int(action_t.item()), log_prob, entropy, value.squeeze(0)

    def update(self, log_prob, entropy, value, reward, next_state, done):
        with torch.no_grad():
            next_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, next_value = self.model(next_t)
            target = torch.tensor([reward], dtype=torch.float32, device=self.device)
            if not done:
                target = target + self.gamma * next_value.squeeze(0)

        advantage = target - value
        actor_loss = -log_prob * advantage.detach()
        critic_loss = F.mse_loss(value, target)
        loss = actor_loss + critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


class REAWS:
    """
    Paper-aligned REAWS reproduction for DAG workflow scheduling.

    The paper uses one global actor-critic scheduler to select a datacenter
    and one local actor-critic scheduler per datacenter to select a server.
    The implementation below preserves that hierarchy and trains online in
    the existing simulator. The minimization objective in Eq. (16) is converted
    to a maximization reward by negating the local time/energy cost.
    """

    def __init__(self, datacenter_count=10, alpha=0.5, is_training=True):
        self.datacenter_count = datacenter_count
        self.alpha = alpha
        self.is_training = is_training
        self.global_agent = None
        self.local_agents = {}
        self.max_hosts_per_dc = 1
        self.global_state_dim = 0
        self.local_state_dim = 0
        self._last_dc_count = 0

    def placement(self, tasks_to_schedule, env, init_state):
        self._ensure_agents(env)
        task_host_pairs = {}
        hosts = env.get_hosts()
        ready_tasks = env.isqualified(tasks_to_schedule)
        datacenters = self._build_datacenters(hosts)
        cur_step_reward = 0.0

        for task in ready_tasks:
            valid_dc_actions = [
                dc["id"] for dc in datacenters
                if any(host.placement_possible(task) for host in dc["hosts"])
            ]
            if not valid_dc_actions:
                continue

            global_state = self._global_state(task, datacenters, env)
            dc_action, g_logp, g_entropy, g_value = self.global_agent.select_action(
                global_state, valid_dc_actions, self.is_training
            )
            dc = next(dc for dc in datacenters if dc["id"] == dc_action)

            valid_host_actions = [
                idx for idx, host in enumerate(dc["hosts"]) if host.placement_possible(task)
            ]
            if not valid_host_actions:
                continue

            local_state = self._local_state(task, dc, env)
            host_action, l_logp, l_entropy, l_value = self.local_agents[dc["id"]].select_action(
                local_state, valid_host_actions, self.is_training
            )
            host = dc["hosts"][host_action]

            local_reward, exec_time, energy = self._local_reward(task, host, env)
            green_reward = self._green_surplus(dc, env) / max(self._dc_full_power(dc), 1.0)
            global_reward = green_reward + local_reward
            cur_step_reward += global_reward

            env.task_assignment(task, host)
            task.end_time = max(task.end_time, env.current_time + exec_time)
            task_host_pairs[task] = host

            next_datacenters = self._build_datacenters(hosts)
            next_global_state = self._global_state(task, next_datacenters, env)
            next_local_state = self._local_state(task, dc, env)
            done = env.if_done()

            if self.is_training:
                self.local_agents[dc["id"]].update(
                    l_logp, l_entropy, l_value, local_reward, next_local_state, done
                )
                self.global_agent.update(
                    g_logp, g_entropy, g_value, global_reward, next_global_state, done
                )

            logging.info(
                "REAWS task=%s dc=%s host=%s local_reward=%.6f green_reward=%.6f T=%.6f E=%.6f",
                task.task_name,
                dc["id"],
                host.host_id,
                local_reward,
                green_reward,
                exec_time,
                energy,
            )

        undeployed_tasks = [task for task in tasks_to_schedule if task not in task_host_pairs]
        info = {
            "assign_num": len(task_host_pairs),
            "task_host_pairs": task_host_pairs,
            "undeployed_tasks": undeployed_tasks,
        }
        return None, cur_step_reward, False, info

    def save(self, base_path="./saved_model/reaws_model/"):
        if self.global_agent is None:
            return
        self.global_agent.save(os.path.join(base_path, "global_actor_critic.pth"))
        for dc_id, agent in self.local_agents.items():
            agent.save(os.path.join(base_path, f"local_{dc_id}_actor_critic.pth"))

    def _ensure_agents(self, env):
        datacenters = self._build_datacenters(env.hosts)
        dc_count = len(datacenters)
        self.max_hosts_per_dc = max(len(dc["hosts"]) for dc in datacenters)
        global_state_dim = 3 * dc_count + 2
        local_state_dim = 2 * self.max_hosts_per_dc + 2

        if (
            self.global_agent is not None
            and self.global_state_dim == global_state_dim
            and self.local_state_dim == local_state_dim
            and self._last_dc_count == dc_count
        ):
            return

        self.global_state_dim = global_state_dim
        self.local_state_dim = local_state_dim
        self._last_dc_count = dc_count
        self.global_agent = _MaskedActorCriticAgent(global_state_dim, dc_count)
        self.local_agents = {
            dc["id"]: _MaskedActorCriticAgent(local_state_dim, self.max_hosts_per_dc)
            for dc in datacenters
        }

    def _build_datacenters(self, hosts):
        dc_count = max(1, min(self.datacenter_count, len(hosts)))
        datacenters = [{"id": idx, "hosts": []} for idx in range(dc_count)]
        for host in hosts:
            datacenters[host.host_id % dc_count]["hosts"].append(host)
        return [dc for dc in datacenters if dc["hosts"]]

    def _global_state(self, task, datacenters, env):
        full_powers = [max(self._dc_full_power(dc), 1.0) for dc in datacenters]
        max_speed = max([self._dc_avg_speed(dc) for dc in datacenters] + [1.0])
        surplus = [self._green_surplus(dc, env) / fp for dc, fp in zip(datacenters, full_powers)]
        speeds = [self._dc_avg_speed(dc) / max_speed for dc in datacenters]
        utils = [self._dc_utilization(dc) for dc in datacenters]
        task_feat = self._task_resource_feature(task)
        return np.array(surplus + speeds + utils + task_feat, dtype=np.float32)

    def _local_state(self, task, dc, env):
        max_speed = max([host.cpu_speed for host in dc["hosts"]] + [1.0])
        speeds = [host.cpu_speed / max_speed for host in dc["hosts"]]
        utils = [self._host_utilization(host) for host in dc["hosts"]]
        pad = self.max_hosts_per_dc - len(dc["hosts"])
        speeds.extend([0.0] * pad)
        utils.extend([0.0] * pad)
        task_feat = self._task_resource_feature(task)
        return np.array(speeds + utils + task_feat, dtype=np.float32)

    def _task_resource_feature(self, task):
        return [
            float(task.plan_cpu) / 8000.0,
            float(task.plan_mem) / 128.0,
        ]

    def _dc_avg_speed(self, dc):
        return float(np.mean([host.cpu_speed for host in dc["hosts"]])) if dc["hosts"] else 0.0

    def _dc_utilization(self, dc):
        if not dc["hosts"]:
            return 0.0
        return float(np.mean([self._host_utilization(host) for host in dc["hosts"]]))

    def _host_utilization(self, host):
        used = host.max_resource.astype(float) - np.array([host.cpu, host.mem, host.gpu], dtype=float)
        return float(np.mean(np.clip(used / np.maximum(host.max_resource.astype(float), 1.0), 0.0, 1.0)))

    def _green_surplus(self, dc, env):
        return self._green_power(dc, env.current_time) - self._dc_current_power(dc)

    def _green_power(self, dc, current_time):
        dc_id = dc["id"]
        host_count = len(dc["hosts"])
        day_pos = (current_time % 144) / 144.0
        phase = dc_id / max(float(self._last_dc_count), 1.0)
        solar = max(0.0, math.sin(math.pi * ((day_pos + phase) % 1.0)))
        wind = 0.35 + 0.15 * math.sin(2.0 * math.pi * (day_pos + phase * 0.5))
        renewable_ratio = min(max(0.15 + 0.65 * solar + wind, 0.0), 1.0)
        avg_full = float(np.mean([host.full for host in dc["hosts"]])) if dc["hosts"] else 180.0
        return renewable_ratio * host_count * avg_full

    def _dc_current_power(self, dc):
        return sum(self._host_current_power(host) for host in dc["hosts"])

    def _dc_full_power(self, dc):
        return sum(float(host.full) for host in dc["hosts"])

    def _host_current_power(self, host):
        utilization = self._host_utilization(host)
        if utilization <= 0.0:
            return 0.0
        return float(host.idle + (host.full - host.idle) * utilization)

    def _local_reward(self, task, host, env):
        exec_time = self._task_execution_time(task, host, env)
        energy = self._task_energy(task, host, exec_time, env)
        time_norm = exec_time / 100.0
        energy_norm = energy / 1000.0
        reward = -(self.alpha * time_norm + (1.0 - self.alpha) * energy_norm)
        return reward, exec_time, energy

    def _task_execution_time(self, task, host, env):
        runtime = max(float(task.task_duration) / max(float(host.cpu_speed), 1e-9), 1e-9)
        waiting_time = max(0.0, self._host_next_available_time(host, env.current_time) - env.current_time)
        transfer_time = self._predecessor_transfer_time(task, host, env)
        return runtime + waiting_time + transfer_time

    def _host_next_available_time(self, host, current_time):
        end_times = [task.end_time for task in host.tasks if task.end_time is not None]
        return max([current_time] + end_times)

    def _task_energy(self, task, host, execution_time, env):
        util = self._host_utilization(host)
        active_power = host.idle + (host.full - host.idle) * max(util, 0.01)
        communication_energy = self._predecessor_transfer_energy(task, host, env)
        return execution_time * active_power + communication_energy

    def _predecessor_transfer_time(self, task, host, env):
        delays = []
        for parent_name in task.parent_tasks:
            parent = env.task_dict.get(parent_name)
            if parent is None or parent.assigned_host is None:
                continue
            bandwidth = 1000.0 if self._same_dc(parent.assigned_host, host) else 600.0
            delays.append(float(getattr(task, "communicate_size", 0.0)) / bandwidth)
        return max(delays, default=0.0)

    def _predecessor_transfer_energy(self, task, host, env):
        total = 0.0
        p_comm = 0.02
        for parent_name in task.parent_tasks:
            parent = env.task_dict.get(parent_name)
            if parent is None or parent.assigned_host is None:
                continue
            bandwidth = 1000.0 if self._same_dc(parent.assigned_host, host) else 600.0
            total += float(getattr(task, "communicate_size", 0.0)) / bandwidth * p_comm
        return total

    def _same_dc(self, host_a, host_b):
        dc_count = max(self._last_dc_count, 1)
        return host_a.host_id % dc_count == host_b.host_id % dc_count
