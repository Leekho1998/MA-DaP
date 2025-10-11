import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os

from sim.util.replay_buffers import BasicBuffer
    
class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals
    
class DDQNAgent:
    def __init__(self, obser_shape, action_shape, batch_size=128, load_file='./sim/saved_model/ddqn_model/', max_size=1000):
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.buffer_mlen = 0.01
        self.tau = 0.01
        self.replay_buffer = BasicBuffer(batch_size, max_size=max_size)
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(obser_shape, action_shape).to(self.device)
        self.target_model = DQN(obser_shape, action_shape).to(self.device)

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

        self.batch_size = batch_size    
        self.update_step = 0

        self.load_file = load_file

    def get_action(self, state, action_space, is_training=True, eps=0.20):
        if np.random.randn() < eps:
            return np.random.choice(action_space)

        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())

        return action, None

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones, _, _ = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        curr_Q = self.model.forward(states).gather(1, actions)
        next_Q = self.target_model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q
        
        loss = F.mse_loss(curr_Q, expected_Q.detach())

        return loss

    def update(self):
        batch = self.replay_buffer.sample()
        loss = self.compute_loss(batch)
        # print("Loss: ", loss.item())
        self.update_step += 1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


    def save(self, ifbest=False):
        os.makedirs(self.load_file, exist_ok=True)
        if ifbest:
            torch.save(self.model.state_dict(), self.load_file+"/best_model.pth")
        else:
            torch.save(self.model.state_dict(), self.load_file+"/model.pth")
        # print("save model to {}".format(self.load_file))

    def load(self, ifbest=True):
        if os.path.exists(self.load_file):
            if ifbest:
                self.model.load_state_dict(torch.load(self.load_file+"/best_model.pth"))
            else:
                self.model.load_state_dict(torch.load(self.load_file+"/model.pth"))
            print("load model from {}".format(self.load_file))
        else:
            print("Model not found.")