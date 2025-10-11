import logging

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os
import copy  # 添加copy模块

from util.replay_buffers import BasicBuffer


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
        return self.fc(state)


class DQNAgent:
    def __init__(self, obser_shape, action_shape, batch_size=128,
                 load_file='./sim/saved_model/dqn_model/', max_size=1000,
                 gamma=0.99, lr=3e-4, target_update_freq=10):
        self.gamma = gamma
        self.learning_rate = lr
        self.buffer_mlen = max_size
        self.replay_buffer = BasicBuffer(batch_size, max_size=max_size)
        self.target_update_freq = target_update_freq  # 目标网络更新频率

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(obser_shape, action_shape).to(self.device)
        # 添加目标网络
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.MSE_loss = nn.MSELoss()

        self.batch_size = batch_size
        self.update_step = 0
        self.load_file = load_file

    def get_action(self, state, action_space, is_training=True, eps=0.2):
        # 处理action_space为整数的情况
        if isinstance(action_space, int):
            action_space = range(action_space)

        if is_training and np.random.random() < eps:
            #print("q-vals random")
            #logging.info(f"q-vals random")
            return np.random.choice(action_space), None


        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            qvals = self.model(state)
        #print(f"q-vals {qvals.cpu().numpy()}")
        #logging.info(f"q-vals {qvals.cpu().numpy()}")
        action = np.argmax(qvals.cpu().numpy())
        return int(action), None

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones, _, _ = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)  # 确保dones在正确设备上

        # 当前Q值
        curr_Q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 使用目标网络计算下一个状态的Q值
        with torch.no_grad():
            next_Q = self.model(next_states)
            max_next_Q = torch.max(next_Q, 1)[0]
            # 考虑终止状态
            expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q

        return self.MSE_loss(curr_Q, expected_Q)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # 确保有足够样本

        self.model.train()
        batch = self.replay_buffer.sample()
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

        self.update_step += 1

        # 更新目标网络（硬更新）
        if self.update_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, model_type="model"):
        os.makedirs(self.load_file, exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_step': self.update_step
        }, os.path.join(self.load_file, f"{model_type}.pth"))

    def load(self, model_type="model"):
        path = os.path.join(self.load_file, f"{model_type}.pth")
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model'])
            self.target_model.load_state_dict(checkpoint['target_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.update_step = checkpoint['update_step']
            print(f"Loaded model from {path}")
        else:
            print("Model not found.")