import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os

from sim.util.replay_buffers import BasicBuffer

import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
    
class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q    

class TD3Agent:
    def __init__(self, state_dim, action_dim, batch_size=128, load_file='./sim/saved_model/td3_model/', learning_rate=3e-4, gamma=0.99, max_size=1000):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_mlen = max_size
        self.replay_buffer = BasicBuffer(batch_size, max_size=max_size)

        self.max_action = action_dim-1
        self.noise_clip = 0.5
        self.policy_noise = 0.2
        self.policy_delay = 2
        self.tau = 0.005
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.batch_size = batch_size    
        self.update_step = 0

        self.load_file = load_file

    def get_action(self, state, action_space, is_training=True, eps=0.20):
        if np.random.random() < eps:
            return np.random.choice(action_space)
        
        state = torch.tensor(state.reshape(1, -1)).float().to(self.device)
        return self.actor(state).cpu().data.numpy().flatten(), None

    def update(self):
        for i in range(20):  # num_iteration源码是100000
            batch = self.replay_buffer.sample()
            states, actions, rewards, next_states, dones, _, _ = batch
            state = torch.FloatTensor(states).to(self.device)
            action = torch.LongTensor(actions).reshape(-1, 1).to(self.device)
            reward = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
            next_state = torch.FloatTensor(next_states).to(self.device)
            done = torch.FloatTensor(dones).reshape(-1, 1).to(self.device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            print(f'Loss/Q1_loss: {loss_Q1.item()}')

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            print(f'Loss/Q2_loss: {loss_Q1.item()}')
            # Delayed policy updates:
            
            if i % self.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                print(f'Loss/actor_loss: {actor_loss.item()}')
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- self.tau) * target_param.data) + self.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

        self.update_step += 1

    def save(self, ifbest=False):
        os.makedirs(self.load_file, exist_ok=True)
        if ifbest:
            torch.save(self.model.state_dict(), self.load_file+"best_model.pth")
            # print("save model to {}".format(self.load_file+"best_model.pth"))
        else:
            torch.save(self.model.state_dict(), self.load_file+"model.pth")
            # print("save model to {}".format(self.load_file+"model.pth"))


    def load(self, ifbest=True):
        if ifbest and os.path.exists(self.load_file+"best_model.pth"):
            self.model.load_state_dict(torch.load(self.load_file+"best_model.pth"))
            print("load model from {}".format(self.load_file+"best_model.pth"))
        elif os.path.exists(self.load_file+"model.pth"):
            self.model.load_state_dict(torch.load(self.load_file+"model.pth"))
            print("load model from {}".format(self.load_file+"model.pth"))
        else:
            print("Model not found.")