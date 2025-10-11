import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, max_size):
        self.max_size = max_size
        self.state_dim = state_dim
        self.s = np.zeros((max_size, state_dim))
        self.a = np.zeros((max_size, 1))
        self.a_logprob = np.zeros((max_size, 1))
        self.r = np.zeros((max_size, 1))
        self.s_ = np.zeros((max_size, state_dim))
        self.dw = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        self.count = 0

    # state, action, (log,) reward, next_state, (dw,) mdone
    def push(self, s, a, r, s_, done, a_logprob, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def reset(self):
        self.s = np.zeros((self.max_size, self.state_dim))
        self.a = np.zeros((self.max_size, 1))
        self.a_logprob = np.zeros((self.max_size, 1))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, self.state_dim))
        self.dw = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))
        self.count = 0

    def get_rewards(self):
        rewards = []
        for re in self.r:
            rewards.append(re)
        return rewards
    
    def reset_rewards(self, new_rewards):
        for i in range(len(new_rewards)):
            self.r[i] = new_rewards[i]

    def sample(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done

    def __len__(self):
        return self.count