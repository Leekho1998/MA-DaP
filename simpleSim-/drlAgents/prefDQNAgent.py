import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os
import torch.nn.functional as F

from util.replay_buffers import BasicBuffer

    
class PrefDQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(PrefDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # =============注意力=============
        hyper_input_dim = 2
        hyper_hidden_embd_dim = 256
        self.embedding_dim = 128
        self.embd_dim = 2
        self.hyper_output_dim = 5 * self.embd_dim  # 10
        self.head_num = 8
        self.qkv_dim = 16

        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)  # 2 -> 256
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True) # 256 -> 256
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True) # 256 -> 10
        
        self.hyper_Wq_first = nn.Linear(self.embd_dim, self.embedding_dim * self.head_num * self.qkv_dim, bias=False) 
        self.hyper_Wq_last = nn.Linear(self.embd_dim, self.embedding_dim * self.head_num * self.qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, self.embedding_dim * self.head_num * self.qkv_dim, bias=False)
        self.hyper_Wv = nn.Linear(self.embd_dim, self.embedding_dim * self.head_num * self.qkv_dim, bias=False)
        # out_dim: 8 * 16 * 128=16384
        self.hyper_multi_head_combine = nn.Linear(self.embd_dim, self.head_num * self.qkv_dim * self.embedding_dim, bias=False)
        # ==============================
        
        # self.fc1 = nn.Sequential(
        #     nn.Linear(self.input_dim[0], 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(256, self.output_dim)
        # )
        self.fc1 = nn.Linear(self.input_dim[0], 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, self.output_dim)

        # 解决图被释放的问题
        # 或者# 将 multi_head_combine_para 定义为 nn.Parameter
        # self.multi_head_combine_para = nn.Parameter(torch.Tensor(self.head_num * self.qkv_dim, self.embedding_dim))
        # nn.init.xavier_uniform_(self.multi_head_combine_para)  # 初始化参数

        # 然后在assign那里
        # with torch.no_grad():
        #     self.multi_head_combine_para.copy_(self.hyper_multi_head_combine(mid_embd[4 * self.embd_dim: 5 * self.embd_dim]).reshape(self.head_num * self.qkv_dim, self.embedding_dim))

        # 这样forward时
        # mh_atten_out = F.linear(out1, self.multi_head_combine_para)

    def assign(self, pref):
        # 用于和多头注意力的输出进行线性变换        
        hyper_embd = self.hyper_fc1(pref) # 2 -> 256
        hyper_embd = self.hyper_fc2(hyper_embd) # 256 -> 256
        mid_embd = self.hyper_fc3(hyper_embd) # 256 -> 10
        
        self.Wq_first_para = self.hyper_Wq_first(mid_embd[:self.embd_dim]).reshape(self.embedding_dim, self.head_num * self.qkv_dim)
        self.Wq_last_para = self.hyper_Wq_last(mid_embd[self.embd_dim:2 * self.embd_dim]).reshape(self.embedding_dim, self.head_num * self.qkv_dim)
        self.Wk_para = self.hyper_Wk(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(self.embedding_dim, self.head_num * self.qkv_dim)
        self.Wv_para = self.hyper_Wv(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(self.embedding_dim, self.head_num * self.qkv_dim)
        self.multi_head_combine_para = self.hyper_multi_head_combine(mid_embd[4 * self.embd_dim: 5 * self.embd_dim]).reshape(self.head_num * self.qkv_dim, self.embedding_dim)
        # self.multi_head_combine_para的输入维度是2，输出维度是16384

    def forward(self, state):
        out1 = self.fc1(state)
        # out1输入是batch_size, 256,每个节点是128维的向量 变成batch_size, 128
        # batch_size,128 x 128, 128 -> batch_size, 128
        mh_atten_out = F.linear(out1, self.multi_head_combine_para.clone().detach())
        mh_atten_out = self.relu(mh_atten_out)
        qvals = self.fc2(out1)
        return qvals
    
class PrefDQNAgent:
    def __init__(self, obser_shape, action_shape, batch_size=128, load_file='./sim/saved_model/dqn_model/', max_size=1000, pref=[0.5, 0.5]):
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.buffer_mlen = max_size
        self.replay_buffer = BasicBuffer(batch_size, max_size=max_size)
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = PrefDQN(obser_shape, action_shape).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.MSE_loss = nn.MSELoss()

        self.batch_size = batch_size    
        self.update_step = 0

        self.load_file = load_file

    def set_pref(self, pref):
        self.model.assign(torch.FloatTensor(pref).to(self.device))

    def get_action(self, state, action_space, is_training=True, eps=0.20):
        if np.random.random() < eps: # random policy
            return np.random.choice(action_space), None
        # greedy policy
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.model.forward(state)
        # if l is not None:
        #     qvals = qvals[:,:l]
        #     qvals = torch.clamp(qvals, 1e-8, 1.0)
        #     qvals = qvals / qvals.sum(dim=1, keepdim=True)
        action = np.argmax(qvals.cpu().detach().numpy()) # action范围0-19
        
        return int(action), None

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones, _, _ = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self):
        batch = self.replay_buffer.sample()
        loss = self.compute_loss(batch)
        # print("Loss: ", loss.item())
        self.update_step += 1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, model_type="model"):  # model_type: best_model, best_reward_model, model
        os.makedirs(self.load_file, exist_ok=True)
        torch.save(self.model.state_dict(), self.load_file+"{}.pth".format(model_type))


    def load(self, model_type="model"):  # model_type: best_model, best_reward_model, model
        if os.path.exists(self.load_file+"{}.pth".format(model_type)):
            self.model.load_state_dict(torch.load(self.load_file+"{}.pth".format(model_type)))
            print("load model from {}".format(self.load_file+"{}.pth".format(model_type)))
        else:
            print("Model not found.")