import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os

from util.replay_buffers import BasicBuffer

import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
    

# The network of the actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        a_prob = F.softmax(self.l2(s), dim=1)
        return a_prob


# The network of the critic
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, 1)

    def forward(self, s):
        s = F.relu(self.l1(s))
        v_s = self.l2(s)
        return v_s



class A2CAgent:
    def __init__(self, state_dim, action_dim, batch_size=1, load_file='./a2c_model/', max_size=1):
        self.lr = 5e-4
        self.gamma = 0.99
        self.buffer_mlen = max_size
        self.replay_buffer = BasicBuffer(batch_size=batch_size, max_size=max_size)
        self.action_dim = action_dim

        self.I = 1
        self.hidden_width = 256

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, self.hidden_width)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(state_dim, self.hidden_width)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.batch_size = batch_size    
        self.update_step = 0

        self.load_file = load_file

    def get_action(self, state, action_space, is_training=True, eps=0.20):
        # if is_training and np.random.random() < eps:
        #     return np.random.choice(action_space)
        
        s = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
        prob_weights = self.actor(s).detach().numpy().flatten()  # probability distribution(numpy)
        if not is_training:  # We use the deterministic policy during the evaluating
            a = np.argmax(prob_weights)  # Select the action with the highest probability
            return a, None
        else:  # We use the stochastic policy during the training
            # 其实就是用到eps的那个策略
            a = np.random.choice(range(self.action_dim), p=prob_weights)  # Sample the action according to the probability distribution
            return a, None

    def update(self):
        batch = self.replay_buffer.sample()
        states, actions, rewards, next_states, dones, _, _ = batch
        for s, a, r, s_, dw in zip(states, actions, rewards, next_states, dones):
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
            s_ = torch.unsqueeze(torch.tensor(s_, dtype=torch.float), 0)
            v_s = self.critic(s).flatten()  # v(s)
            v_s_ = self.critic(s_).flatten()  # v(s')

            with torch.no_grad():  # td_target has no gradient
                td_target = r + self.gamma * (1 - dw) * v_s_

            # Update actor
            log_pi = torch.log(self.actor(s).flatten()[a])  # log pi(a|s)
            actor_loss = -self.I * ((td_target - v_s).detach()) * log_pi  # Only calculate the derivative of log_pi
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic
            critic_loss = (td_target - v_s) ** 2  # Only calculate the derivative of v(s)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.I *= self.gamma  # Represent the gamma^t in th policy gradient theorem

        self.update_step += 1

    def save(self, ifbest=False):
        os.makedirs(self.load_file, exist_ok=True)
        if ifbest:
            torch.save(self.actor.state_dict(), self.load_file+"actor_best_model.pth")
            torch.save(self.critic.state_dict(), self.load_file+"critic_best_model.pth")
            # print("save model to {}".format(self.load_file+"(best)"))
        else:
            torch.save(self.actor.state_dict(), self.load_file+"actor_model.pth")
            torch.save(self.critic.state_dict(), self.load_file+"critic_model.pth")
            # print("save model to {}".format(self.load_file+"()"))


    def load(self, ifbest=True):
        if ifbest and os.path.exists(self.load_file):
            self.actor.load_state_dict(torch.load(self.load_file+"actor_best_model.pth"))
            self.critic.load_state_dict(torch.load(self.load_file+"critic_best_model.pth"))
            print("load model from {}".format(self.load_file+"(best)"))
        elif os.path.exists(self.load_file):
            self.actor.load_state_dict(torch.load(self.load_file+"actor_model.pth"))
            self.critic.load_state_dict(torch.load(self.load_file+"critic_model.pth"))
            print("load model from {}".format(self.load_file+"()"))
        else:
            print("Model not found.")