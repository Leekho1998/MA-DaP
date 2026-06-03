import os

import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.signal
from HPCSimPickJobs import *


class Buffer():
    def __init__(self):
        self.buffer_num = 0
        self.states = []
        self.actions = []
        self.masks = []
        self.log_probs = []
        self.Returns = []
        self.advantages = []

    def clear_buffer(self):
        self.buffer_num = 0
        self.states = []
        self.actions = []
        self.masks = []
        self.log_probs = []
        self.Returns = []
        self.advantages = []

    def store_buffer(self, state, mask, action, log_prob, Return, advantage, nums):
        self.buffer_num = self.buffer_num + nums
        self.states.extend(state)
        self.masks.extend(mask)
        self.actions.extend(action)
        self.log_probs.extend(log_prob)
        self.Returns.extend(Return)
        self.advantages.extend(advantage)


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[],device=None):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            # print(self.masks.shape)
            # print(logits.shape)
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(-1)

class ActorNet(nn.Module):
    def __init__(self, num_inputs1, featureNum1, num_inputs2, featureNum2, num_inputs3, featureNum3):
        super(ActorNet, self).__init__()
        self.d_model = 128

        self.num_inputs1 = num_inputs1
        self.featureNum1 = featureNum1
        self.num_inputs2 = num_inputs2
        self.featureNum2 = featureNum2
        self.num_inputs3 = num_inputs3
        self.featureNum3 = featureNum3

        self.JobEncoder = nn.Sequential(
            nn.Linear(self.featureNum1, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        job = x[:, :self.num_inputs1, :self.featureNum1]
        job = self.JobEncoder(job)
        con = job
        con =  self.decoder1(con)

        logits = con.squeeze(dim=-1)

        return logits


class CriticNet(nn.Module):

    def __init__(self, num_inputs1, featureNum1, num_inputs2, featureNum2, num_inputs3, featureNum3):
        super(CriticNet, self).__init__()
        self.d_model = 128

        self.num_inputs1 = num_inputs1
        self.featureNum1 = featureNum1
        self.num_inputs2 = num_inputs2
        self.featureNum2 = featureNum2
        self.num_inputs3 = num_inputs3
        self.featureNum3 = featureNum3

        self.JobEncoder = nn.Sequential(
            nn.Linear(self.featureNum1, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.GreenEncoder = nn.Sequential(
            nn.Linear(self.featureNum3, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.RunningJobEncoder = nn.Sequential(
            nn.Linear(self.featureNum2, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.hidden = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear((self.num_inputs1 + self.num_inputs2 + self.num_inputs3) * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        job = x[:, :self.num_inputs1, :self.featureNum1]
        run = x[:, self.num_inputs1:self.num_inputs1 + self.num_inputs2, :self.featureNum2]
        green = x[:, self.num_inputs1 + self.num_inputs2:self.num_inputs1 + self.num_inputs2 + self.num_inputs3,
                :self.featureNum3]
        green = self.GreenEncoder(green)
        job = self.JobEncoder(job)
        run = self.RunningJobEncoder(run)
        con = torch.cat([job, run, green], dim=1)

        con = self.hidden(con)
        con = self.flatten(con)
        value = self.out(con)
        return value


class PPO():
    def __init__(self, batch_size=10, inputNum_size=[], featureNum_size=[],
                 device='cpu'):
        super(PPO, self).__init__()
        self.num_inputs1 = inputNum_size[0]
        self.num_inputs2 = inputNum_size[1]
        self.num_inputs3 = inputNum_size[2]

        self.featureNum1 = featureNum_size[0]
        self.featureNum2 = featureNum_size[1]
        self.featureNum3 = featureNum_size[2]

        self.device = device
        self.actor_net = ActorNet(
            self.num_inputs1, self.featureNum1, self.num_inputs2, self.featureNum2, self.num_inputs3,
            self.featureNum3).to(self.device)
        self.critic_net = CriticNet(
            self.num_inputs1, self.featureNum1, self.num_inputs2, self.featureNum2, self.num_inputs3,
            self.featureNum3).to(self.device)
        self.batch_size = batch_size
        self.gamma = 1
        self.lam = 0.97

        self.states = []
        self.log_probs = []
        self.rewards_seq = []
        self.actions = []
        self.values = []
        self.masks = []
        self.entropys = []
        self.buffer = Buffer()

        self.ppo_update_time = 8
        self.clip_param = 0.2
        self.max_grad_norm = 0.5

        self.entropy_coefficient = 0

        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), lr=0.0001,eps=1e-6)
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=0.0005,eps=1e-6)

    def act(self, state, mask):
        logits = self.actor_net(state)

        value = self.critic_net(state)
        dist_bin = CategoricalMasked(logits=logits, masks=mask[:, :self.num_inputs1],device=self.device)
        id = dist_bin.sample()

        log_prob = dist_bin.log_prob(id)
        return id, log_prob, value

    def act1(self, state, mask, action):
        logits = self.actor_net(state)

        dist_bin = CategoricalMasked(logits=logits, masks=mask[:, :self.num_inputs1],device=self.device)
        log_prob = dist_bin.log_prob(action)
        entropy = dist_bin.entropy()
        return log_prob, entropy

    def normalize(self, advantages):
        nor_advantages = (advantages - torch.mean(advantages)) / (
                torch.std(advantages) + 1e-9)
        return nor_advantages

    def remember(self, state, value, log_prob, action, reward, mask, device):
        self.rewards_seq.append(reward)
        self.states.append(state.to("cpu"))
        self.log_probs.append(log_prob.to("cpu"))
        self.values.append(value.to("cpu"))
        self.actions.append(action.to("cpu"))
        self.masks.append(mask.to("cpu"))

    def clear_memory(self):
        self.rewards_seq = []
        self.states = []
        self.log_probs = []
        self.values = []
        self.actions = []
        self.masks = []

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_val=0):
        rews = np.append(np.array(self.rewards_seq), last_val)
        values = torch.cat(self.values, dim=0)
        values = values.squeeze(dim=-1)
        vals = np.append(np.array(values.cpu()), last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = self.discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        ret = self.discount_cumsum(rews, self.gamma)[:-1]
        # ret=adv+vals[:-1]
        return adv, ret

    def storeIntoBuffter(self, reward):
        advantages, returns = self.finish_path(reward)
        returns = returns.tolist()
        advantages = advantages.tolist()

        self.buffer.store_buffer(self.states, self.masks, self.actions, self.log_probs, returns, advantages,
                                 len(self.states))

    def compute_value_loss(self, states, returns):
        state_values = self.critic_net(states)
        state_values = torch.squeeze(state_values, dim=1)

        # Calculate value loss using F.mse_loss
        value_loss = F.mse_loss(state_values, returns)
        return value_loss

    def compute_actor_loss(self,
                           states,
                           masks,
                           actions,
                           advantages,
                           old_log_probs
                           ):

        log_probs, entropy = self.act1(states, masks, actions)
        # Compute the policy loss
        logratio = log_probs - old_log_probs
        ratio = torch.exp(logratio)

        surr1 = ratio * advantages
        clip_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
        surr2 = clip_ratio * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))  # MAX->MIN descent
        entropy_loss = torch.mean(entropy)

        total_loss = policy_loss - self.entropy_coefficient * entropy_loss

        return total_loss, policy_loss, entropy_loss

    def train(self):
        states = torch.cat(self.buffer.states, dim=0)
        masks = torch.cat(self.buffer.masks, dim=0)
        actions = torch.cat(self.buffer.actions, dim=0)
        log_probs = torch.cat(self.buffer.log_probs, dim=0)
        returns = torch.tensor(self.buffer.Returns)
        advantages = torch.tensor(self.buffer.advantages)
        advantages = self.normalize(advantages)
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer.states))), self.batch_size, False):
                index_tensor = torch.tensor(index)
                sampled_states = torch.index_select(states, dim=0, index=index_tensor).to(self.device)
                sampled_masks = torch.index_select(masks, dim=0, index=index_tensor).to(self.device)
                sampled_actions = torch.index_select(actions, dim=0, index=index_tensor).to(self.device)
                sampled_log_probs = torch.index_select(log_probs, dim=0, index=index_tensor).to(self.device)
                sampled_returns = torch.index_select(returns, dim=0, index=index_tensor).to(self.device)
                sampled_advantages = torch.index_select(advantages, dim=0, index=index_tensor).to(self.device)
                action_loss, polic_loss, entropy_loss = self.compute_actor_loss(sampled_states, sampled_masks,
                                                                                sampled_actions, sampled_advantages,
                                                                                sampled_log_probs)
                value_loss = self.compute_value_loss(sampled_states, sampled_returns)

                self.actor_optimizer.zero_grad()
                self.critic_net_optimizer.zero_grad()

                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()


    def save_using_model_name(self, model_name_path):
        if not os.path.exists(model_name_path):
            # 如果目录不存在，则创建它
            os.makedirs(model_name_path)
        torch.save(self.actor_net.state_dict(), model_name_path + "actor.pkl")
        torch.save(self.critic_net.state_dict(),
                   model_name_path + "critic.pkl")

    def load_using_model_name(self, model_name_path):
        self.actor_net.load_state_dict(
            torch.load(model_name_path + "actor.pkl"))
        self.critic_net.load_state_dict(
            torch.load(model_name_path + "critic.pkl"))

    def eval_action(self,o,mask):
        with torch.no_grad():
            o = o.reshape(1, MAX_QUEUE_SIZE + run_win + green_win, JOB_FEATURES)
            state = torch.FloatTensor(o).to(self.device)
            mask = np.array(mask).reshape(1, MAX_QUEUE_SIZE + run_win + green_win)
            mask = torch.FloatTensor(mask).to(self.device)
            logits = self.actor_net(state)

            value = self.critic_net(state)
            dist_bin = CategoricalMasked(logits=logits, masks=mask[:, :self.num_inputs1], device=self.device)
            ac = dist_bin.sample()

        return ac

def train(workload,backfill):
    seed = 0
    epochs = 300
    traj_num = 100
    env = HPCEnv(backfill=1)
    env.seed(seed)
    current_dir = os.getcwd()
    workload_name = workload
    workload_file = os.path.join(current_dir, "./data/" + workload_name + ".swf")
    env.my_init(workload_file=workload_file)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    inputNum_size = [MAX_QUEUE_SIZE, run_win, green_win]
    featureNum_size = [JOB_FEATURES, RUN_FEATURE, GREEN_FEATURE]
    ppo = PPO(batch_size=256, inputNum_size=inputNum_size,
              featureNum_size=featureNum_size, device=device)
    for epoch in range(epochs):
        o, r, d, ep_ret, ep_len, show_ret, sjf, f1, greenRwd = env.reset(), 0, False, 0, 0, 0, 0, 0, 0
        t = 0
        epoch_reward = 0
        green_reward=0
        wait_reward=0
        while True:
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                    lst.append(0)
                elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                    lst.append(0)
                else:
                    lst.append(1)
            for i in range(run_win):
                lst.append(0)
            for i in range(green_win):
                lst.append(0)

            with torch.no_grad():
                o = o.reshape(1, MAX_QUEUE_SIZE + run_win + green_win, JOB_FEATURES)
                state = torch.FloatTensor(o).to(device)
                mask = np.array(lst).reshape(1, MAX_QUEUE_SIZE + run_win + green_win)
                mask = torch.FloatTensor(mask).to(device)
                ind, log_prob, value = ppo.act(state, mask)
            ppo.remember(state, value, log_prob, ind, r, mask, device)

            o, r, d, r2, sjf_t, f1_t,running_num,greenRwd = env.step(ind.item(),0)
            ep_ret += r
            ep_len += 1
            show_ret += r2
            sjf += sjf_t
            f1 += f1_t
            green_reward+=greenRwd
            wait_reward+=r
            epoch_reward +=  eta*r+greenRwd

            if d:
                t += 1
                ppo.storeIntoBuffter(eta*r+greenRwd)
                ppo.clear_memory()
                o, r, d, ep_ret, ep_len, show_ret, sjf, f1, greenRwd = env.reset(), 0, False, 0, 0, 0, 0, 0, 0
                if t >= traj_num:
                    break

        ppo.train()
        with open('MaskablePPO_' + workload_name + '.csv', mode='a',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerow([float(epoch_reward / traj_num),float(green_reward / traj_num),float(wait_reward / traj_num)])
        ppo.buffer.clear_buffer()

    ppo.save_using_model_name(workload_name + '/MaskablePPO')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='lublin_256')
    parser.add_argument('--backfill', type=int, default=0)
    args = parser.parse_args()
    train(args.workload, args.backfill)