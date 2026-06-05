import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class _ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        encoded = self.encoder(state)
        return self.actor(encoded), self.critic(encoded)


class _PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        epochs=4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.epochs = epochs
        self.buffer = []

    def select_action(self, state, action_mask, is_training=True):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(state_t)
        mask_t = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        masked_logits = logits.masked_fill(~mask_t, -1e9)
        dist = Categorical(logits=masked_logits)
        action = dist.sample() if is_training else torch.argmax(masked_logits, dim=-1)
        return int(action.item()), float(dist.log_prob(action).item()), float(value.squeeze().item())

    def push(self, state, action, log_prob, value, reward, done):
        self.buffer.append(
            {
                "state": np.asarray(state, dtype=np.float32),
                "action": int(action),
                "log_prob": float(log_prob),
                "value": float(value),
                "reward": float(reward),
                "done": bool(done),
            }
        )

    def update(self):
        if len(self.buffer) < 2:
            return
        rewards = np.array([item["reward"] for item in self.buffer], dtype=np.float32)
        values = np.array([item["value"] for item in self.buffer] + [0.0], dtype=np.float32)
        dones = np.array([item["done"] for item in self.buffer], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for idx in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[idx]
            delta = rewards[idx] + self.gamma * values[idx + 1] * non_terminal - values[idx]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[idx] = gae
        returns = advantages + values[:-1]
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(np.stack([item["state"] for item in self.buffer]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([item["action"] for item in self.buffer], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([item["log_prob"] for item in self.buffer], dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        for _ in range(self.epochs):
            logits, value = self.model(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            policy_loss = -torch.min(
                ratio * adv_t,
                torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_t,
            ).mean()
            value_loss = F.mse_loss(value.squeeze(-1), ret_t)
            entropy = dist.entropy().mean()
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        self.buffer.clear()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


class GPMARL:
    """
    Paper-aligned GP-MARL reproduction.

    GP-MARL uses decentralized training and decentralized execution: each PPO
    scheduler agent owns a partition of machines and chooses a combined
    machine/job-slot action. The job slots are shared across agents, matching
    the DeepRM-like multi-agent environment described in the paper.
    """

    def __init__(
        self,
        num_agents=3,
        machines_per_agent=2,
        job_slot_size=5,
        local_observation=True,
        local_reward=True,
        is_training=True,
    ):
        self.num_agents = num_agents
        self.machines_per_agent = machines_per_agent
        self.job_slot_size = job_slot_size
        self.local_observation = local_observation
        self.local_reward = local_reward
        self.is_training = is_training
        self.agents = []
        self.state_dim = 0
        self.action_dim = machines_per_agent * job_slot_size

    def placement(self, tasks_to_schedule, env, init_state):
        self._ensure_agents(env)
        ready_tasks = [task for task in tasks_to_schedule if task.start_time is None]
        ready_tasks.sort(key=lambda task: (task.submit_time, task.task_name))
        job_slots = ready_tasks[: self.job_slot_size]
        task_host_pairs = {}
        if not job_slots:
            return None, 0.0, env.if_done(), {
                "assign_num": 0,
                "task_host_pairs": task_host_pairs,
                "undeployed_tasks": ready_tasks,
            }

        partitions = self._partition_hosts(env.hosts)
        total_reward = 0.0
        done = env.if_done()
        for agent_id, hosts in enumerate(partitions):
            state = self._state(agent_id, partitions, job_slots, ready_tasks, env)
            mask = self._action_mask(hosts, job_slots)
            if not np.any(mask):
                continue
            action, log_prob, value = self.agents[agent_id].select_action(state, mask, self.is_training)
            machine_idx = action // self.job_slot_size
            slot_idx = action % self.job_slot_size
            if machine_idx >= len(hosts) or slot_idx >= len(job_slots):
                continue
            task = job_slots[slot_idx]
            host = hosts[machine_idx]
            if task.start_time is not None or not host.placement_possible(task):
                continue
            env.task_assignment(task, host)
            task_host_pairs[task] = host
            reward = self._reward(agent_id, partitions, ready_tasks, env)
            total_reward += reward
            if self.is_training:
                self.agents[agent_id].push(state, action, log_prob, value, reward, done)
            logging.info(
                "GP-MARL agent=%s action=%s task=%s host=%s reward=%.6f",
                agent_id,
                action,
                task.task_name,
                host.host_id,
                reward,
            )

        undeployed_tasks = [task for task in ready_tasks if task.start_time is None]
        return None, total_reward, env.if_done(), {
            "assign_num": len(task_host_pairs),
            "task_host_pairs": task_host_pairs,
            "undeployed_tasks": undeployed_tasks,
        }

    def finish_episode(self):
        if self.is_training:
            for agent in self.agents:
                agent.update()

    def save(self, base_path="./saved_model/gp_marl_model/"):
        for idx, agent in enumerate(self.agents):
            agent.save(os.path.join(base_path, f"agent_{idx}_ppo.pth"))

    def _ensure_agents(self, env):
        state_dim = self._state_dim(env)
        if self.agents and self.state_dim == state_dim:
            return
        self.state_dim = state_dim
        self.agents = [_PPOAgent(state_dim, self.action_dim) for _ in range(self.num_agents)]

    def _state_dim(self, env):
        host_count = self.machines_per_agent if self.local_observation else len(env.hosts)
        return host_count * 6 + self.job_slot_size * 5 + 1

    def _partition_hosts(self, hosts):
        ordered = sorted(hosts, key=lambda host: host.host_id)
        needed = self.num_agents * self.machines_per_agent
        padded = ordered[:needed]
        partitions = []
        for agent_id in range(self.num_agents):
            start = agent_id * self.machines_per_agent
            partitions.append(padded[start:start + self.machines_per_agent])
        return partitions

    def _state(self, agent_id, partitions, job_slots, ready_tasks, env):
        hosts = partitions[agent_id] if self.local_observation else [host for part in partitions for host in part]
        host_features = []
        for host in hosts:
            host_features.extend(
                [
                    host.cpu / max(host.max_resource[0], 1),
                    host.mem / max(host.max_resource[1], 1),
                    host.gpu / max(host.max_resource[2], 1),
                    host.cpu_speed / 10.0,
                    len(host.tasks) / 10.0,
                    self._host_next_available(host, env.current_time) / 144.0,
                ]
            )
        expected_hosts = self.machines_per_agent if self.local_observation else self.num_agents * self.machines_per_agent
        while len(host_features) < expected_hosts * 6:
            host_features.extend([0.0] * 6)

        slot_features = []
        for task in job_slots:
            slot_features.extend(
                [
                    max(env.current_time - task.submit_time, 0) / 144.0,
                    task.task_duration / 100.0,
                    task.plan_cpu / 8000.0,
                    task.plan_mem / 128.0,
                    task.plan_gpu / 800.0,
                ]
            )
        while len(slot_features) < self.job_slot_size * 5:
            slot_features.extend([0.0] * 5)
        backlog = max(len(ready_tasks) - self.job_slot_size, 0) / 100.0
        return np.array(host_features + slot_features + [backlog], dtype=np.float32)

    def _action_mask(self, hosts, job_slots):
        mask = np.zeros(self.action_dim, dtype=bool)
        for machine_idx, host in enumerate(hosts):
            for slot_idx, task in enumerate(job_slots):
                action = machine_idx * self.job_slot_size + slot_idx
                mask[action] = task.start_time is None and host.placement_possible(task)
        return mask

    def _reward(self, agent_id, partitions, ready_tasks, env):
        waiting_penalty = sum(1.0 / max(float(task.task_duration), 1.0) for task in ready_tasks[: self.job_slot_size])
        backlog_penalty = sum(1.0 / max(float(task.task_duration), 1.0) for task in ready_tasks[self.job_slot_size:])
        if self.local_reward:
            managed_hosts = partitions[agent_id]
        else:
            managed_hosts = [host for part in partitions for host in part]
        running = [
            task
            for host in managed_hosts
            for task in host.tasks
            if task.end_time is not None and task.end_time > env.current_time
        ]
        running_penalty = sum(1.0 / max(float(task.end_time - env.current_time), 1.0) for task in running)
        return -(waiting_penalty + backlog_penalty + running_penalty)

    def _host_next_available(self, host, current_time):
        end_times = [task.end_time for task in host.tasks if task.end_time is not None]
        return max([float(current_time)] + end_times)
