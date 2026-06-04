import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class _GASMarlActorCritic(nn.Module):
    def __init__(self, wait_dim, run_dim, green_dim, wait_window, delay_dim):
        super().__init__()
        self.wait_window = wait_window
        self.wait_encoder = nn.Sequential(
            nn.Linear(wait_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.run_encoder = nn.Sequential(
            nn.Linear(run_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.green_encoder = nn.Sequential(
            nn.Linear(green_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.job_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, wait_window))
        self.selected_job = nn.Sequential(nn.Linear(wait_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU())
        self.delay_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, delay_dim))
        self.critic = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 1))

    def encode(self, wait_state, run_state, green_state):
        wait_nodes = self.wait_encoder(wait_state)
        wait_emb = wait_nodes.mean(dim=1)
        run_emb = self.run_encoder(run_state).mean(dim=1)
        green_emb = self.green_encoder(green_state).mean(dim=1)
        return wait_nodes, wait_emb, run_emb, green_emb

    def forward(self, wait_state, run_state, green_state, selected_index=None):
        wait_nodes, wait_emb, run_emb, green_emb = self.encode(wait_state, run_state, green_state)
        job_logits = self.job_head(wait_emb)
        if selected_index is None:
            selected_index = torch.argmax(job_logits, dim=-1)
        batch_index = torch.arange(wait_state.size(0), device=wait_state.device)
        selected_feat = self.selected_job(wait_state[batch_index, selected_index])
        delay_in = torch.cat([wait_emb, run_emb, green_emb, selected_feat], dim=-1)
        delay_logits = self.delay_head(delay_in)
        value = self.critic(torch.cat([wait_emb, run_emb, green_emb], dim=-1))
        return job_logits, delay_logits, value


class _GASMarlPPO:
    def __init__(
        self,
        wait_window,
        run_window,
        delay_dim,
        lr=3e-4,
        gamma=1.0,
        gae_lambda=0.97,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        batch_size=64,
        epochs=4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wait_window = wait_window
        self.run_window = run_window
        self.delay_dim = delay_dim
        self.model = _GASMarlActorCritic(8, 4, 2, wait_window, delay_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.batch_size = batch_size
        self.epochs = epochs
        self.buffer = []

    def select_action(self, state, job_mask, delay_masks, is_training=True):
        wait, run, green = self._state_tensors(state)
        job_logits, _, value = self.model(wait, run, green)
        masked_job_logits = job_logits.masked_fill(~torch.tensor(job_mask, dtype=torch.bool, device=self.device).unsqueeze(0), -1e9)
        job_dist = Categorical(logits=masked_job_logits)
        job_action = job_dist.sample() if is_training else torch.argmax(masked_job_logits, dim=-1)
        _, delay_logits, value = self.model(wait, run, green, job_action)
        selected_delay_mask = delay_masks[int(job_action.item())]
        masked_delay_logits = delay_logits.masked_fill(~torch.tensor(selected_delay_mask, dtype=torch.bool, device=self.device).unsqueeze(0), -1e9)
        delay_dist = Categorical(logits=masked_delay_logits)
        delay_action = delay_dist.sample() if is_training else torch.argmax(masked_delay_logits, dim=-1)
        log_prob = job_dist.log_prob(job_action) + delay_dist.log_prob(delay_action)
        entropy = job_dist.entropy() + delay_dist.entropy()
        return (
            int(job_action.item()),
            int(delay_action.item()),
            float(log_prob.item()),
            float(value.squeeze().item()),
            float(entropy.item()),
        )

    def push(self, state, job_action, delay_action, log_prob, value, reward, done):
        self.buffer.append(
            {
                "wait": np.asarray(state["wait"], dtype=np.float32),
                "run": np.asarray(state["run"], dtype=np.float32),
                "green": np.asarray(state["green"], dtype=np.float32),
                "job_action": int(job_action),
                "delay_action": int(delay_action),
                "log_prob": float(log_prob),
                "value": float(value),
                "reward": float(reward),
                "done": bool(done),
            }
        )

    def set_last_reward(self, reward, done=True):
        if self.buffer:
            self.buffer[-1]["reward"] = float(reward)
            self.buffer[-1]["done"] = bool(done)

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

        wait = torch.tensor(np.stack([item["wait"] for item in self.buffer]), dtype=torch.float32, device=self.device)
        run = torch.tensor(np.stack([item["run"] for item in self.buffer]), dtype=torch.float32, device=self.device)
        green = torch.tensor(np.stack([item["green"] for item in self.buffer]), dtype=torch.float32, device=self.device)
        job_actions = torch.tensor([item["job_action"] for item in self.buffer], dtype=torch.long, device=self.device)
        delay_actions = torch.tensor([item["delay_action"] for item in self.buffer], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([item["log_prob"] for item in self.buffer], dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        for _ in range(self.epochs):
            job_logits, delay_logits, value = self.model(wait, run, green, job_actions)
            job_dist = Categorical(logits=job_logits)
            delay_dist = Categorical(logits=delay_logits)
            new_log_probs = job_dist.log_prob(job_actions) + delay_dist.log_prob(delay_actions)
            entropy = job_dist.entropy().mean() + delay_dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            policy_loss = -torch.min(
                ratio * adv_t,
                torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_t,
            ).mean()
            value_loss = F.mse_loss(value.squeeze(-1), ret_t)
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        self.buffer.clear()

    def _state_tensors(self, state):
        wait = torch.tensor(state["wait"], dtype=torch.float32, device=self.device).unsqueeze(0)
        run = torch.tensor(state["run"], dtype=torch.float32, device=self.device).unsqueeze(0)
        green = torch.tensor(state["green"], dtype=torch.float32, device=self.device).unsqueeze(0)
        return wait, run, green

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


class GASMARL:
    """
    Paper-aligned GAS-MARL reproduction for HPC-style batch job scheduling.

    GAS-MARL is the learning part: a PPO actor-critic with two sequential
    sub-actions, job selection and delay decision. Green-Backfilling remains a
    heuristic policy and is triggered by delay/resource reservation, as in the
    paper.
    """

    def __init__(
        self,
        wait_window=32,
        run_window=16,
        eta=0.003,
        brown_threshold=50000.0,
        is_training=True,
    ):
        self.wait_window = wait_window
        self.run_window = run_window
        self.eta = eta
        self.brown_threshold = brown_threshold
        self.is_training = is_training
        self.delay_candidates = [0, "run1", "run2", "run3", "run4", "run5", 1, 2, 4, 6, 8, 10, 12]
        self.agent = _GASMarlPPO(wait_window, run_window, len(self.delay_candidates))
        self.last_metrics = {"renewable_util": 0.0, "avg_bsd": 0.0}

    def placement(self, tasks_to_schedule, env, init_state):
        waiting = [task for task in tasks_to_schedule if task.start_time is None]
        waiting.sort(key=lambda task: (task.submit_time, task.task_name))
        visible = waiting[: self.wait_window]
        task_host_pairs = {}
        if not visible:
            return None, 0.0, env.if_done(), {
                "assign_num": 0,
                "task_host_pairs": task_host_pairs,
                "undeployed_tasks": waiting,
            }

        state = self._state(env, visible)
        job_mask = np.zeros(self.wait_window, dtype=bool)
        job_mask[: len(visible)] = True
        delay_masks = np.zeros((self.wait_window, len(self.delay_candidates)), dtype=bool)
        for idx, task in enumerate(visible):
            delay_masks[idx] = self._delay_mask(task, env)
        job_action, delay_action, log_prob, value, _entropy = self.agent.select_action(
            state, job_mask, delay_masks, self.is_training
        )
        selected = visible[job_action]
        delay_mask = self._delay_mask(selected, env)
        if not delay_mask[delay_action]:
            delay_action = 0

        delay = self._delay_slots(delay_action, env)
        if delay > 0:
            selected.exc_time = env.current_time + delay
            self._green_backfill(waiting, selected, env, task_host_pairs, env.current_time + delay)
        else:
            if not self._assign(selected, env, task_host_pairs):
                reserve_time = self._earliest_start(selected, env)
                self._green_backfill(waiting, selected, env, task_host_pairs, reserve_time)
            else:
                self._green_backfill(waiting, selected, env, task_host_pairs, env.current_time + 1)

        done = env.if_done()
        reward = self._terminal_reward(env) if done else 0.0
        if self.is_training:
            self.agent.push(state, job_action, delay_action, log_prob, value, reward, done)
        undeployed_tasks = [task for task in waiting if task.start_time is None and task.exc_time is None]
        return None, reward, done, {
            "assign_num": len(task_host_pairs),
            "task_host_pairs": task_host_pairs,
            "undeployed_tasks": undeployed_tasks,
        }

    def finish_episode(self, env):
        reward = self._terminal_reward(env)
        self.agent.set_last_reward(reward, done=True)
        if self.is_training:
            self.agent.update()
        return reward

    def save(self, base_path="./saved_model/gas_marl_model/"):
        self.agent.save(os.path.join(base_path, "ppo_actor_critic.pth"))

    def _assign(self, task, env, task_host_pairs):
        host = self._best_host(task, env)
        if host is None:
            return False
        env.task_assignment(task, host)
        task_host_pairs[task] = host
        logging.info("GAS-MARL assign task=%s host=%s time=%s", task.task_name, host.host_id, env.current_time)
        return True

    def _green_backfill(self, waiting, blocked_task, env, task_host_pairs, limit_time):
        candidates = [task for task in waiting if task is not blocked_task and task.start_time is None and task.exc_time is None]
        candidates.sort(key=lambda task: max(task.task_duration, 1) * max(task.plan_cpu, 1) * self._task_power(task, env))
        for task in candidates:
            host = self._best_host(task, env)
            if host is None:
                continue
            finish = env.current_time + task.task_duration / max(host.cpu_speed, 1e-9)
            brown_energy = self._brown_energy_if_start_now(task, env)
            if finish < limit_time and brown_energy < self.brown_threshold:
                self._assign(task, env, task_host_pairs)

    def _state(self, env, visible):
        wait_rows = np.zeros((self.wait_window, 8), dtype=np.float32)
        for idx, task in enumerate(visible[: self.wait_window]):
            power = self._task_power(task, env)
            brown_energy = self._brown_energy_if_start_now(task, env)
            total_energy = max(power * max(float(task.task_duration), 1.0), 1e-9)
            wait_rows[idx] = np.array(
                [
                    max(env.current_time - task.submit_time, 0) / 144.0,
                    float(task.task_duration) / 100.0,
                    float(task.plan_cpu) / 8000.0,
                    power / 1000.0,
                    power / max(float(task.plan_cpu), 1.0) / 100.0,
                    1.0 if brown_energy > 0 else 0.0,
                    min(brown_energy / total_energy, 1.0),
                    1.0 if self._best_host(task, env) is not None else 0.0,
                ],
                dtype=np.float32,
            )

        running = []
        for host in env.hosts:
            for task in host.tasks:
                if task.end_time is not None and task.end_time > env.current_time:
                    running.append((task.end_time, task, host))
        running.sort(key=lambda item: item[0])
        run_rows = np.zeros((self.run_window, 4), dtype=np.float32)
        for idx, (_, task, _host) in enumerate(running[: self.run_window]):
            power = self._task_power(task, env)
            run_rows[idx] = np.array(
                [
                    float(task.plan_cpu) / 8000.0,
                    power / 1000.0,
                    power / max(float(task.plan_cpu), 1.0) / 100.0,
                    max(task.end_time - env.current_time, 0.0) / 100.0,
                ],
                dtype=np.float32,
            )

        green_rows = np.zeros((24, 2), dtype=np.float32)
        for idx in range(24):
            green_rows[idx] = np.array([
                (24 - idx) / 24.0,
                self._renewable_power(env.current_time + idx) / max(self._cluster_full_power(env), 1.0),
            ])
        return {"wait": wait_rows, "run": run_rows, "green": green_rows}

    def _delay_mask(self, task, env):
        mask = np.zeros(len(self.delay_candidates), dtype=bool)
        remaining = max(int(env.remaining_decline(task)), 0)
        for idx in range(len(mask)):
            mask[idx] = self._delay_slots(idx, env) <= remaining
        mask[0] = True
        return mask

    def _delay_slots(self, delay_action, env):
        choice = self.delay_candidates[delay_action]
        if choice == 0:
            return 0
        if isinstance(choice, str) and choice.startswith("run"):
            n = int(choice[3:])
            end_times = sorted(
                task.end_time for host in env.hosts for task in host.tasks
                if task.end_time is not None and task.end_time > env.current_time
            )
            if len(end_times) >= n:
                return min(int(np.ceil(end_times[n - 1] - env.current_time)), 12)
            return min(n, 12)
        return int(choice)

    def _best_host(self, task, env):
        candidates = [host for host in env.hosts if host.placement_possible(task)]
        if not candidates:
            return None
        return min(candidates, key=lambda host: (self._incremental_power(host, task), host.host_id))

    def _earliest_start(self, task, env):
        if self._best_host(task, env) is not None:
            return env.current_time
        end_times = [
            task.end_time for host in env.hosts for task in host.tasks
            if task.end_time is not None and task.end_time > env.current_time
        ]
        return min(end_times, default=env.current_time + 1)

    def _brown_energy_if_start_now(self, task, env):
        duration = max(int(np.ceil(task.task_duration)), 1)
        task_power = self._task_power(task, env)
        brown = 0.0
        for step in range(duration):
            renewable = self._renewable_power(env.current_time + step)
            current_power = self._cluster_current_power(env)
            brown += max(current_power + task_power - renewable, 0.0)
        return brown

    def _terminal_reward(self, env):
        renewable_util = self._renewable_utilization(env)
        avg_bsd = self._avg_bounded_slowdown(env)
        self.last_metrics = {"renewable_util": renewable_util, "avg_bsd": avg_bsd}
        return renewable_util - self.eta * avg_bsd

    def _renewable_utilization(self, env):
        if env.current_time <= 0:
            return 0.0
        total = 0.0
        renewable_used = 0.0
        for t in range(int(np.ceil(env.current_time)) + 1):
            power = self._cluster_power_at(env, t)
            total += power
            renewable_used += min(power, self._renewable_power(t))
        return renewable_used / max(total, 1e-9)

    def _avg_bounded_slowdown(self, env):
        values = []
        for task in env.tasks:
            if task.start_time is None or task.end_time is None:
                continue
            wait = max(task.start_time - task.submit_time, 0.0)
            run = max(task.end_time - task.start_time, 1e-9)
            values.append(max((wait + run) / max(10.0, run), 1.0))
        return float(np.mean(values)) if values else 0.0

    def _task_power(self, task, env):
        return env.marginal_task_power(task) if hasattr(env, "marginal_task_power") else float(task.plan_cpu)

    def _incremental_power(self, host, task):
        cpu = task.plan_cpu / max(host.max_resource[0], 1)
        mem = task.plan_mem / max(host.max_resource[1], 1)
        gpu = task.plan_gpu / max(host.max_resource[2], 1)
        return 5.0 * cpu + 2.0 * mem + 3.0 * gpu

    def _cluster_current_power(self, env):
        return sum(env.energyConsumptionPerHost) if hasattr(env, "energyConsumptionPerHost") else 0.0

    def _cluster_full_power(self, env):
        return sum(float(host.full) for host in env.hosts)

    def _cluster_power_at(self, env, time_slot):
        power = 0.0
        for task in env.tasks:
            if task.start_time is not None and task.end_time is not None and task.start_time <= time_slot < task.end_time:
                power += self._task_power(task, env)
        return power

    def _renewable_power(self, time_slot):
        day_pos = (time_slot % 144) / 144.0
        solar = max(0.0, np.sin(np.pi * day_pos))
        wind = 0.35 + 0.15 * np.sin(2.0 * np.pi * day_pos)
        return 5000.0 * min(max(0.65 * solar + wind, 0.0), 1.0)
