import logging
import os
import torch as T
import torch.nn.functional as F
import numpy as np
from sim.mydrl.networks import ActorNetwork, CriticNetwork
from sim.util.replay_buffers import BasicBuffer

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

# https://github.com/155469wxh/AI/blob/925a45f133bd59e92fe9483b52c49d2939bbc776/%E6%96%87%E4%BB%B6%E5%90%88%E9%9B%86/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/DDPG-main/train.py

class DDPGAgent:
    def __init__(self, state_dim, action_dim, batch_size=128, load_file='./sim/saved_model/ddpg_model/', max_size=1000):
        self.alpha = 0.0003
        self.beta = 0.0003
        self.gamma = 0.99
        self.tau = 0.005
        self.action_noise = 0.1
        self.buffer_mlen = max_size

        self.actor = ActorNetwork(alpha=self.alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=400, fc2_dim=300)
        self.target_actor = ActorNetwork(alpha=self.alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=400, fc2_dim=300)
        self.critic = CriticNetwork(beta=self.beta, state_dim=state_dim, action_dim=action_dim,
                                    fc1_dim=400, fc2_dim=300)
        self.target_critic = CriticNetwork(beta=self.beta, state_dim=state_dim, action_dim=action_dim,
                                           fc1_dim=400, fc2_dim=300)

        self.replay_buffer = BasicBuffer(batch_size, max_size=max_size)

        self.update_network_parameters(tau=1.0)

        self.batch_size = batch_size
        self.update_step = 0

        self.load_file = load_file

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic_params, target_critic_params in zip(self.critic.parameters(),
                                                       self.target_critic.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)

    def get_action(self, observation, action_space, is_training=True, eps=0.2):
        # if is_training and np.random.random() < eps:
        #     return np.random.choice(action_space)
        
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(device)
        a_logprob = self.actor.forward(state).squeeze()

        if is_training:
            noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=T.float).to(device)
            a_logprob = T.clamp(a_logprob+noise, -1, 1)
            self.actor.train()

        a_logprob = a_logprob.detach().cpu().numpy()
        return np.argmax(a_logprob), a_logprob

    def update(self):
        batch = self.replay_buffer.sample()
        states, _, reward, states_, terminals, a_logprob, _ = batch
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(a_logprob, dtype=T.float).to(device)
        rewards_tensor = T.tensor(reward, dtype=T.float).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            q_ = self.target_critic.forward(next_states_tensor, next_actions_tensor).view(-1)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_
        q = self.critic.forward(states_tensor, actions_tensor).view(-1)

        critic_loss = F.mse_loss(q, target.detach())
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        new_actions_tensor = self.actor.forward(states_tensor)
        actor_loss = -T.mean(self.critic(states_tensor, new_actions_tensor))
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def save(self, ifbest=False):
        os.makedirs(self.load_file+'/Actor', exist_ok=True)
        os.makedirs(self.load_file+'/Critic', exist_ok=True)
        if ifbest:
            episode = "best_model"
        else:
            episode = "model"
        T.save(self.actor.state_dict(), self.load_file + 'DDPG_actor_{}.pth'.format(episode))
        # print('Saving actor network successfully!')
        T.save(self.target_actor.state_dict(), self.load_file +
                                          'DDPG_target_actor_{}.pth'.format(episode))
        # print('Saving target_actor network successfully!')
        T.save(self.critic.state_dict(), self.load_file + 'DDPG_critic_{}.pth'.format(episode))
        # print('Saving critic network successfully!')
        T.save(self.target_critic.state_dict(), self.load_file +
                                           'DDPG_target_critic_{}.pth'.format(episode))
        # print('Saving target critic network successfully!')

    def load(self, ifbest=True):
        if os.path.exists(self.load_file):
            if ifbest:
                episode = "best_model"
            else:
                episode = "model"
            self.actor.load_state_dict(T.load(self.load_file + 'DDPG_actor_{}.pth'.format(episode)))
            logging.info('Loading actor network successfully!')
            self.target_actor.load_state_dict(T.load(self.load_file +
                                            'DDPG_target_actor_{}.pth'.format(episode)))
            logging.info('Loading target_actor network successfully!')
            self.critic.load_state_dict(T.load(self.load_file + 'DDPG_critic_{}'.format(episode)))
            logging.info('Loading critic network successfully!')
            self.target_critic.load_state_dict(T.load(self.load_file +
                                            'DDPG_target_critic_{}'.format(episode)))
            logging.info('Loading target critic network successfully!')
        else:
            logging.info("Model not found.")
