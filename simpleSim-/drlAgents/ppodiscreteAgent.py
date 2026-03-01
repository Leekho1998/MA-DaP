import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical

from util.replay_buffers import BasicBuffer
from util.replay_buffers_ppo import ReplayBuffer

NMAX = 10
NODE_F = 4
DAG_DIM = NMAX * NODE_F + NMAX * NMAX + NMAX   # 40 + 100 + 10 = 150
GNN_OUT = 32
# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width=64, use_tanh=True, use_orthogonal_init=True):
        super().__init__()
        assert state_dim > DAG_DIM, f"state_dim must include DAG_DIM={DAG_DIM} at the end"
        self.raw_dim = state_dim - DAG_DIM

        # ---- Forward GAT params (parents -> v) ----
        self.Wf = nn.Linear(NODE_F, GNN_OUT, bias=False)
        self.af_src = nn.Parameter(torch.empty(GNN_OUT))
        self.af_dst = nn.Parameter(torch.empty(GNN_OUT))

        # ---- Backward GAT params (children -> v) ----
        self.Wb = nn.Linear(NODE_F, GNN_OUT, bias=False)
        self.ab_src = nn.Parameter(torch.empty(GNN_OUT))
        self.ab_dst = nn.Parameter(torch.empty(GNN_OUT))

        # ---- Gate to fuse gf/gb ----
        self.gate = nn.Sequential(
            nn.Linear(GNN_OUT * 2, GNN_OUT),
            nn.ReLU(),
            nn.Linear(GNN_OUT, 1),
            nn.Sigmoid()
        )

        # ---- Your original MLP (input dim changed: raw + fused_graph_emb) ----
        self.fc1 = nn.Linear(self.raw_dim + GNN_OUT, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][use_tanh]

        # init
        nn.init.xavier_uniform_(self.Wf.weight)
        nn.init.xavier_uniform_(self.Wb.weight)
        nn.init.xavier_uniform_(self.af_src.view(1, -1))
        nn.init.xavier_uniform_(self.af_dst.view(1, -1))
        nn.init.xavier_uniform_(self.ab_src.view(1, -1))
        nn.init.xavier_uniform_(self.ab_dst.view(1, -1))

        if use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def _parse_dag(self, dag_part):
        B = dag_part.size(0)
        x_flat = dag_part[:, :NMAX*NODE_F]
        a_flat = dag_part[:, NMAX*NODE_F:NMAX*NODE_F + NMAX*NMAX]
        m_flat = dag_part[:, -NMAX:]

        X = x_flat.view(B, NMAX, NODE_F)        # [B,N,F]
        A = a_flat.view(B, NMAX, NMAX)          # [B,N,N] parent->child
        mask = m_flat.view(B, NMAX, 1)          # [B,N,1]
        # self-loop
        I = torch.eye(NMAX, device=dag_part.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = (A + I).clamp(max=1.0)
        return X, A_hat, mask

    def _gat_one_pass(self, X, A_hat, mask, W, a_src, a_dst):
        """
        Compute node outputs H: [B,N,O] using adjacency A_hat (i->j).
        We normalize over incoming neighbors for each target j (softmax over i).
        """
        B = X.size(0)
        Wh = W(X)                                # [B,N,O]

        f_src = (Wh * a_src.view(1, 1, -1)).sum(-1)  # [B,N]
        f_dst = (Wh * a_dst.view(1, 1, -1)).sum(-1)  # [B,N]
        e = f_src.unsqueeze(2) + f_dst.unsqueeze(1)  # [B,N,N]
        e = F.leaky_relu(e, 0.2)

        neg_inf = torch.tensor(-1e9, device=X.device)
        e = torch.where(A_hat > 0, e, neg_inf)

        alpha = torch.softmax(e, dim=1)          # incoming softmax over i
        H = torch.matmul(alpha.transpose(1, 2), Wh)  # [B,N,O]
        H = torch.relu(H)

        # mask out padded nodes for later pooling
        H = H * mask
        return H

    def _masked_pool(self, H, mask):
        denom = mask.sum(dim=1).clamp(min=1.0)   # [B,1]
        g = H.sum(dim=1) / denom                 # [B,O]
        return g

    def _bi_encoder(self, dag_part):
        X, A_hat, mask = self._parse_dag(dag_part)

        # Forward: parents -> v uses A_hat (parent->child)
        Hf = self._gat_one_pass(X, A_hat, mask, self.Wf, self.af_src, self.af_dst)
        gf = self._masked_pool(Hf, mask)         # [B,O]

        # Backward: children -> v uses A_hat^T (child->parent)
        A_rev = A_hat.transpose(1, 2)
        Hb = self._gat_one_pass(X, A_rev, mask, self.Wb, self.ab_src, self.ab_dst)
        gb = self._masked_pool(Hb, mask)         # [B,O]

        # Gate fuse
        gate = self.gate(torch.cat([gf, gb], dim=-1))  # [B,1]
        g = gate * gf + (1 - gate) * gb
        return g

    def forward(self, s):
        raw = s[:, :self.raw_dim]
        dag = s[:, self.raw_dim:]                # last 150 dims
        g = self._bi_encoder(dag)

        z = torch.cat([raw, g], dim=-1)
        z = self.activate_func(self.fc1(z))
        z = self.activate_func(self.fc2(z))
        a_prob = torch.softmax(self.fc3(z), dim=1)
        return a_prob

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width=64, use_tanh=True, use_orthogonal_init=True):
        super().__init__()
        assert state_dim > DAG_DIM, f"state_dim must include DAG_DIM={DAG_DIM} at the end"
        self.raw_dim = state_dim - DAG_DIM

        self.Wf = nn.Linear(NODE_F, GNN_OUT, bias=False)
        self.af_src = nn.Parameter(torch.empty(GNN_OUT))
        self.af_dst = nn.Parameter(torch.empty(GNN_OUT))

        self.Wb = nn.Linear(NODE_F, GNN_OUT, bias=False)
        self.ab_src = nn.Parameter(torch.empty(GNN_OUT))
        self.ab_dst = nn.Parameter(torch.empty(GNN_OUT))

        self.gate = nn.Sequential(
            nn.Linear(GNN_OUT * 2, GNN_OUT),
            nn.ReLU(),
            nn.Linear(GNN_OUT, 1),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(self.raw_dim + GNN_OUT, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][use_tanh]

        nn.init.xavier_uniform_(self.Wf.weight)
        nn.init.xavier_uniform_(self.Wb.weight)
        nn.init.xavier_uniform_(self.af_src.view(1, -1))
        nn.init.xavier_uniform_(self.af_dst.view(1, -1))
        nn.init.xavier_uniform_(self.ab_src.view(1, -1))
        nn.init.xavier_uniform_(self.ab_dst.view(1, -1))

        if use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def _parse_dag(self, dag_part):
        B = dag_part.size(0)
        x_flat = dag_part[:, :NMAX*NODE_F]
        a_flat = dag_part[:, NMAX*NODE_F:NMAX*NODE_F + NMAX*NMAX]
        m_flat = dag_part[:, -NMAX:]
        X = x_flat.view(B, NMAX, NODE_F)
        A = a_flat.view(B, NMAX, NMAX)
        mask = m_flat.view(B, NMAX, 1)
        I = torch.eye(NMAX, device=dag_part.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = (A + I).clamp(max=1.0)
        return X, A_hat, mask

    def _gat_one_pass(self, X, A_hat, mask, W, a_src, a_dst):
        Wh = W(X)
        f_src = (Wh * a_src.view(1, 1, -1)).sum(-1)
        f_dst = (Wh * a_dst.view(1, 1, -1)).sum(-1)
        e = F.leaky_relu(f_src.unsqueeze(2) + f_dst.unsqueeze(1), 0.2)
        neg_inf = torch.tensor(-1e9, device=X.device)
        e = torch.where(A_hat > 0, e, neg_inf)
        alpha = torch.softmax(e, dim=1)
        H = torch.matmul(alpha.transpose(1, 2), Wh)
        H = torch.relu(H) * mask
        return H

    def _masked_pool(self, H, mask):
        denom = mask.sum(dim=1).clamp(min=1.0)
        return H.sum(dim=1) / denom

    def _bi_encoder(self, dag_part):
        X, A_hat, mask = self._parse_dag(dag_part)
        Hf = self._gat_one_pass(X, A_hat, mask, self.Wf, self.af_src, self.af_dst)
        gf = self._masked_pool(Hf, mask)
        A_rev = A_hat.transpose(1, 2)
        Hb = self._gat_one_pass(X, A_rev, mask, self.Wb, self.ab_src, self.ab_dst)
        gb = self._masked_pool(Hb, mask)
        gate = self.gate(torch.cat([gf, gb], dim=-1))
        return gate * gf + (1 - gate) * gb

    def forward(self, s):
        raw = s[:, :self.raw_dim]
        dag = s[:, self.raw_dim:]
        g = self._bi_encoder(dag)
        z = torch.cat([raw, g], dim=-1)
        z = self.activate_func(self.fc1(z))
        z = self.activate_func(self.fc2(z))
        return self.fc3(z)

class PPO_discrete:
    def __init__(self, state_dim, action_dim, max_size=1000, batch_size=128, load_file='./sim/saved_model/ppo_discrete_model/'):
        self.max_size = max_size
        self.batch_size = batch_size
        self.max_train_steps = int(2e5)
        self.lr_a = 3e-4  # Learning rate of actor
        self.lr_c = 3e-4  # Learning rate of critic
        self.gamma = 0.99  # Discount factor
        self.lamda = 0.95  # GAE parameter
        self.epsilon = 0.2  # PPO clip parameter
        self.K_epochs = 10  # PPO parameter
        self.entropy_coef = 0.01  # Entropy coefficient
        self.set_adam_eps = True
        self.use_grad_clip = True
        self.use_lr_decay = True
        self.use_adv_norm = True

        self.buffer_mlen = max_size
        # self.replay_buffer = BasicBuffer(batch_size, max_size)
        self.replay_buffer = ReplayBuffer(state_dim, max_size)

        self.total_steps = 0
        self.update_step = 0

        self.actor = Actor(state_dim=state_dim, action_dim=action_dim)
        self.critic = Critic(state_dim=state_dim)

        self.load_file = load_file

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = self.actor(s).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def get_action(self, observation, action_space, is_training=True, eps=0.2, l=None):
        self.total_steps += 1  

        # if is_training and np.random.random() < eps:  # 0或者注释掉？？
        #     return np.random.choice(action_space), 0

        observation = torch.unsqueeze(torch.tensor(observation, dtype=torch.float), 0)
        self.actor.eval()
        with torch.no_grad():
            # Categorical的输入是离散概率分布，输出是采样的动作
            probss=self.actor(observation)

            if l is not None:
                probss = probss[:,:l]
                probss = torch.clamp(probss, 1e-8, 1.0)
                probss = probss / probss.sum(dim=1, keepdim=True)

            dist = Categorical(probs=probss)
            a = dist.sample()
            a_logprob = dist.log_prob(a)

        # return a.numpy()[0], a_logprob.numpy()[0]
        return a.numpy()[0], a_logprob.numpy()[0]

    def update(self):
        s, a, a_logprob, r, s_, dw, done = self.replay_buffer.sample()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        self.actor.train()
        self.critic.train()
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.max_size)), self.batch_size, False):
                dist_now = Categorical(probs=self.actor(s[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(batch_size X 1)
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(self.total_steps)


    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    # def save_model(self, path):
    #     torch.save(self.actor, path + '_actor.pth')
    #     torch.save(self.critic, path + '_critic.pth')

    def save(self, model_type="model"):  # model_type: best_model, best_reward_model, model
        os.makedirs(self.load_file, exist_ok=True)
        torch.save(self.actor.state_dict(), self.load_file + "actor_{}.pth".format(model_type))
        torch.save(self.critic.state_dict(), self.load_file + "critic_{}.pth".format(model_type))

    # def load_model(self, path):
    #     self.actor = torch.load(path + '_actor.pth')
    #     self.critic = torch.load(path + '_critic.pth')
    #     self.actor.eval()
    #     self.critic.eval()
    #     print("model has been loaded")

    def load(self, model_type="best_model"):  # model_type: best_model, best_reward_model, model
        if os.path.exists(self.load_file):
            self.actor.load_state_dict(torch.load(self.load_file + "actor_{}.pth".format(model_type)))
            self.critic.load_state_dict(torch.load(self.load_file + "critic_{}.pth".format(model_type)))
            logging.info("Loaded model from {}".format(self.load_file + "actor_{}.pth".format(model_type)))
        else:
            logging.info("Model not found.")
