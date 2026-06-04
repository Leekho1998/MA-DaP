import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class _PPOActorCritic(nn.Module):
    def __init__(self, state_dim, dc_dim, server_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.dc_actor = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, dc_dim))
        self.server_actor = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, server_dim))
        self.critic = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, state):
        enc = self.encoder(state)
        return self.dc_actor(enc), self.server_actor(enc), self.critic(enc)


class _RAPPO:
    def __init__(
        self,
        state_dim,
        dc_dim,
        server_dim,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.1,
        vf_coef=0.5,
        ent_coef=0.01,
        batch_size=64,
        epochs=4,
        lr=3e-4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _PPOActorCritic(state_dim, dc_dim, server_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.batch_size = batch_size
        self.epochs = epochs
        self.buffer = []

    def select_action(self, state, valid_dcs, server_masks, is_training=True):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dc_logits, server_logits, value = self.model(state_t)

        dc_mask = torch.full_like(dc_logits, -1e9)
        dc_mask[0, list(valid_dcs)] = 0.0
        dc_probs = torch.softmax(dc_logits + dc_mask, dim=-1)
        dc_dist = Categorical(probs=dc_probs)
        dc_action = dc_dist.sample() if is_training else torch.argmax(dc_probs, dim=-1)

        server_mask_np = server_masks[int(dc_action.item())]
        server_mask = torch.full_like(server_logits, -1e9)
        valid_servers = np.where(server_mask_np)[0].tolist()
        server_mask[0, valid_servers] = 0.0
        server_probs = torch.softmax(server_logits + server_mask, dim=-1)
        server_dist = Categorical(probs=server_probs)
        server_action = server_dist.sample() if is_training else torch.argmax(server_probs, dim=-1)

        log_prob = dc_dist.log_prob(dc_action) + server_dist.log_prob(server_action)
        entropy = dc_dist.entropy() + server_dist.entropy()
        confidence = float(dc_probs[0, dc_action].item() * server_probs[0, server_action].item())
        return (
            int(dc_action.item()),
            int(server_action.item()),
            float(log_prob.item()),
            float(value.squeeze().item()),
            float(entropy.item()),
            confidence,
        )

    def push(self, state, dc_action, server_action, log_prob, value, reward, done):
        self.buffer.append(
            {
                "state": np.asarray(state, dtype=np.float32),
                "dc_action": int(dc_action),
                "server_action": int(server_action),
                "log_prob": float(log_prob),
                "value": float(value),
                "reward": float(reward),
                "done": bool(done),
            }
        )

    def maybe_update(self):
        if len(self.buffer) < self.batch_size:
            return
        self.update()
        self.buffer.clear()

    def update(self):
        if not self.buffer:
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
        dc_actions = torch.tensor([item["dc_action"] for item in self.buffer], dtype=torch.long, device=self.device)
        server_actions = torch.tensor([item["server_action"] for item in self.buffer], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([item["log_prob"] for item in self.buffer], dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        for _ in range(self.epochs):
            dc_logits, server_logits, value = self.model(states)
            dc_dist = Categorical(logits=dc_logits)
            server_dist = Categorical(logits=server_logits)
            new_log_probs = dc_dist.log_prob(dc_actions) + server_dist.log_prob(server_actions)
            entropy = dc_dist.entropy().mean() + server_dist.entropy().mean()
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

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


class ECMWS:
    """
    Electricity Cost-aware Multiple Workflows Scheduling.

    This is a simulator-level reproduction of the ECMWS framework: CWS workflow
    sequencing, BLDP sub-deadline partitioning, TS3 task sequencing, CCRA
    resource allocation with a RAPPO actor-critic policy, and DARA fallback
    when policy confidence is below the threshold.
    """

    def __init__(
        self,
        alpha=(0.4, 0.2, 0.4),
        beta=1.0,
        bandwidth=1000.0,
        datacenter_count=4,
        conf_thresh=0.2,
        is_training=True,
    ):
        self.alpha1, self.alpha2, self.alpha3 = alpha
        self.beta = beta
        self.bandwidth = bandwidth
        self.datacenter_count = datacenter_count
        self.conf_thresh = conf_thresh
        self.is_training = is_training
        self._cache = {}
        self.agent = None
        self.dc_count = 0
        self.max_servers_per_dc = 0
        self.state_dim = 0

    def placement(self, tasks_to_schedule, env, init_state):
        self._ensure_agent(env)
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
        datacenters = self._build_datacenters(hosts)
        cur_step_reward = 0.0

        for task in ordered_tasks:
            state, dc_action, server_action, host, reward = self._ccra(task, datacenters, env, profiles)
            if host is None:
                continue
            task_host_pairs[task] = host
            env.task_assignment(task, host)
            task.end_time = max(task.end_time, self._task_finish_time(task, host, env.current_time, env))
            cur_step_reward += reward
            if self.is_training and state is not None:
                self.agent.push(
                    state,
                    dc_action,
                    server_action,
                    self._last_log_prob,
                    self._last_value,
                    reward,
                    env.if_done(),
                )
                self.agent.maybe_update()

        undeployed_tasks = [task for task in tasks_to_schedule if task not in task_host_pairs]
        info = {
            "assign_num": len(task_host_pairs),
            "task_host_pairs": task_host_pairs,
            "undeployed_tasks": undeployed_tasks,
        }
        return None, cur_step_reward, False, info

    def save(self, base_path="./saved_model/ecmws_model/"):
        if self.agent is not None:
            self.agent.update()
            self.agent.save(os.path.join(base_path, "rappo_actor_critic.pth"))

    def _ensure_agent(self, env):
        datacenters = self._build_datacenters(env.hosts)
        dc_count = len(datacenters)
        max_servers = max(len(dc["hosts"]) for dc in datacenters)
        state_dim = 10 + (dc_count * 4) + (dc_count * 24) + 7
        if (
            self.agent is not None
            and self.dc_count == dc_count
            and self.max_servers_per_dc == max_servers
            and self.state_dim == state_dim
        ):
            return
        self.dc_count = dc_count
        self.max_servers_per_dc = max_servers
        self.state_dim = state_dim
        self.agent = _RAPPO(state_dim, dc_count, max_servers)

    def _build_datacenters(self, hosts):
        dc_count = max(1, min(self.datacenter_count, len(hosts)))
        datacenters = [{"id": idx, "hosts": []} for idx in range(dc_count)]
        for host in hosts:
            datacenters[host.host_id % dc_count]["hosts"].append(host)
        return [dc for dc in datacenters if dc["hosts"]]

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
            "root_rank": root_rank,
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

    def _ccra(self, task, datacenters, env, profiles):
        server_masks = {}
        valid_dcs = []
        for dc in datacenters:
            mask = np.zeros(self.max_servers_per_dc, dtype=bool)
            for idx, host in enumerate(dc["hosts"]):
                mask[idx] = host.placement_possible(task)
            server_masks[dc["id"]] = mask
            if np.any(mask):
                valid_dcs.append(dc["id"])
        if not valid_dcs:
            return None, None, None, None, 0.0

        state = self._state_vector(task, datacenters, env, profiles)
        dc_action, server_action, log_prob, value, _entropy, confidence = self.agent.select_action(
            state, valid_dcs, server_masks, self.is_training
        )
        self._last_log_prob = log_prob
        self._last_value = value
        dc = next(dc for dc in datacenters if dc["id"] == dc_action)
        chosen = dc["hosts"][server_action] if server_action < len(dc["hosts"]) else None
        if chosen is None or not chosen.placement_possible(task) or confidence < self.conf_thresh:
            chosen = self._dara(task, datacenters, env, profiles)
            if chosen is None:
                return state, dc_action, server_action, None, 0.0

        reward, cost, finish_time, sub_deadline = self._reward(task, chosen, env, profiles)
        logging.info(
            "ECMWS task=%s host=%s confidence=%.6f reward=%.6f cost=%.6f finish=%.6f sub_deadline=%.6f",
            task.task_name,
            chosen.host_id,
            confidence,
            reward,
            cost,
            finish_time,
            sub_deadline,
        )
        return state, dc_action, server_action, chosen, reward

    def _dara(self, task, datacenters, env, profiles):
        sub_deadline = profiles["sub_deadline"].get(task, float(task.submit_time) + float(task.decline))
        earliest_start = self._task_earliest_start(task, env.current_time, env)
        dc_order = sorted(datacenters, key=lambda dc: self._dc_price(dc["id"], int(np.ceil(earliest_start)), env))
        fallback = None
        fallback_ratio = -float("inf")
        for dc in dc_order:
            best = None
            best_cost = float("inf")
            for host in dc["hosts"]:
                if not host.placement_possible(task):
                    continue
                finish = self._task_finish_time(task, host, env.current_time, env)
                cost = self._electricity_cost(task, host, env, int(np.ceil(finish - env.current_time)))
                ratio = float(host.cpu_speed) / max(float(host.full), 1.0)
                if ratio > fallback_ratio:
                    fallback_ratio = ratio
                    fallback = host
                if finish <= sub_deadline and cost < best_cost:
                    best_cost = cost
                    best = host
            if best is not None:
                return best
        return fallback

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

    def _host_utilization(self, host):
        used = host.max_resource.astype(float) - np.array([host.cpu, host.mem, host.gpu], dtype=float)
        return float(np.mean(np.clip(used / np.maximum(host.max_resource.astype(float), 1.0), 0.0, 1.0)))

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

    def _state_vector(self, task, datacenters, env, profiles):
        sub_deadline = profiles["sub_deadline"].get(task, float(task.submit_time) + float(task.decline))
        task_rank = profiles["task_rank"].get(task, 0.0)
        workflow = profiles["workflow"].get(self._workflow_id(task), {})
        task_state = [
            float(task.task_duration) / 100.0,
            float(task.plan_cpu) / 8000.0,
            float(task.plan_mem) / 128.0,
            float(task.plan_gpu) / 800.0,
            float(task.communicate_size) / 10000.0,
            max(sub_deadline - env.current_time, 0.0) / 144.0,
            task_rank / max(workflow.get("root_rank", 1.0), 1.0),
            workflow.get("workflow_rank", 0.0),
            workflow.get("slack", 0.0) / 144.0,
            workflow.get("contention", 0.0) / max(len(env.hosts), 1),
        ]
        server_state = []
        for dc in datacenters:
            utils = [self._host_utilization(host) for host in dc["hosts"]]
            speeds = [float(host.cpu_speed) for host in dc["hosts"]]
            powers = [float(host.full) for host in dc["hosts"]]
            server_state.extend([
                float(np.mean(speeds)) / 10.0 if speeds else 0.0,
                float(np.mean(utils)) if utils else 0.0,
                float(np.mean(powers)) / 1000.0 if powers else 0.0,
                self._dc_price(dc["id"], env.current_time, env),
            ])
        price_state = []
        for dc in datacenters:
            prices = [self._dc_price(dc["id"], env.current_time + hour, env) for hour in range(24)]
            p_min = min(prices)
            p_max = max(prices)
            denom = max(p_max - p_min, 1e-9)
            price_state.extend([(price - p_min) / denom for price in prices])
        params = [
            self.alpha1,
            self.alpha2,
            self.alpha3,
            self.beta / 10.0,
            0.0,
            0.0,
            1.0,
        ]
        return np.array(task_state + server_state + price_state + params, dtype=np.float32)

    def _reward(self, task, host, env, profiles):
        sub_deadline = profiles["sub_deadline"].get(task, float(task.submit_time) + float(task.decline))
        finish_time = self._task_finish_time(task, host, env.current_time, env)
        duration = max(int(np.ceil(finish_time - env.current_time)), 1)
        cost = self._electricity_cost(task, host, env, duration)
        run_span = max(finish_time - env.current_time, 1e-9)
        penalty = 1.0 + max(0.0, finish_time - sub_deadline) / run_span
        return -cost * penalty, cost, finish_time, sub_deadline

    def _task_finish_time(self, task, host, current_time, env):
        start = max(self._task_earliest_start(task, current_time, env), self._host_available_time(host, current_time))
        return start + self._predecessor_transfer_delay(task, host, env) + max(
            float(task.task_duration) / max(float(host.cpu_speed), 1e-9),
            1e-9,
        )

    def _task_earliest_start(self, task, current_time, env):
        parent_finish = []
        for parent_name in task.parent_tasks:
            parent = env.task_dict.get(parent_name)
            if parent is not None and parent.end_time is not None:
                parent_finish.append(parent.end_time)
        return max([float(current_time)] + parent_finish)

    def _host_available_time(self, host, current_time):
        return max([float(current_time)] + [task.end_time for task in host.tasks if task.end_time is not None])

    def _dc_price(self, dc_id, time_slot, env):
        phase = int(144 * dc_id / max(self.dc_count, 1))
        return float(env.price_at(int(time_slot) + phase))
