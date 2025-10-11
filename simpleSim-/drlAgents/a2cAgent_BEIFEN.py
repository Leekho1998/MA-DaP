import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os

from sim.util.replay_buffers import BasicBuffer

import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
    

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value



class A2CAgent:
    def __init__(self, state_dim, action_dim, batch_size=1, load_file='./sim/saved_model/a2c_model/', max_size=1):
        self.lr = 5e-4
        self.gamma = 0.99
        self.buffer_mlen = max_size
        self.replay_buffer = BasicBuffer(batch_size=batch_size, max_size=max_size)
        self.action_dim = action_dim

        self.I = 1
        self.hidden_width = 256

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(state_dim, action_dim, self.hidden_width).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

        self.batch_size = batch_size    
        self.update_step = 0

        self.load_file = load_file

    def get_action(self, state, action_space, is_training=True, eps=0.20):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        dist, value = self.model(state)

        action = dist.sample()
        return int(action.cpu().numpy()), None
    
    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def update(self):
        batch = self.replay_buffer.sample()
        states, actions, rewards, next_states, dones, _, _ = batch
        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0
        last_new_state = next_states[-1]

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, value = self.model(state)

            action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))

        next_state = torch.FloatTensor(last_new_state).unsqueeze(0).to(self.device)
        _, next_value = self.model(next_state)
        returns = self.compute_returns(next_value, rewards, masks)
        
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_step += 1

    def save(self, ifbest=False):
        os.makedirs(self.load_file, exist_ok=True)
        if ifbest:
            torch.save(self.model.state_dict(), self.load_file+"best_model.pth")
            # print("save model to {}".format(self.load_file+"(best)"))
        else:
            torch.save(self.model.state_dict(), self.load_file+"model.pth")
            # print("save model to {}".format(self.load_file+"()"))


    def load(self, ifbest=True):
        if ifbest and os.path.exists(self.load_file):
            self.model.load_state_dict(torch.load(self.load_file+"best_model.pth"))
            print("load model from {}".format(self.load_file+"(best)"))
        elif os.path.exists(self.load_file):
            self.model.load_state_dict(torch.load(self.load_file+"model.pth"))
            print("load model from {}".format(self.load_file+"()"))
        else:
            print("Model not found.")