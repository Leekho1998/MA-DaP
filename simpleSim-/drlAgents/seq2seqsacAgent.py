import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os
import copy

from sim.mydrl.seq_model import Decoder, Encoder, Seq2Seq, init_weights
from sim.util.replay_buffers import BasicBuffer

import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

input_dim = 84  # ['<unk>'：0, '<pad>'：1, '<sos>', '<eos>', '.', 'ein', 'einem', 'in', 'eine', ',']
output_dim = 1   # ['<unk>', '<pad>', '<sos>', '<eos>', 'a', '.', 'in', 'the', 'on', 'man']
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 1
encoder_dropout = 0.5
decoder_dropout = 0.5
teacher_forcing_ratio = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip = 1.0

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size=None, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        
        self.encoder = Encoder(
                            input_dim,
                            encoder_embedding_dim,
                            hidden_dim,
                            n_layers,
                            encoder_dropout,
                        )
        self.decoder = Decoder(
                            output_dim,
                            decoder_embedding_dim,
                            hidden_dim,
                            n_layers,
                            decoder_dropout,
                        )
        self.model = Seq2Seq(self.encoder, self.decoder, device)
        self.model.apply(init_weights)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        action_probs = self.softmax(self.model(state, teacher_forcing_ratio))
        return action_probs
    
    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state, l):
        with torch.no_grad():
            action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
            
        return action.detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size=None, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.encoder = Encoder(
                            input_dim,
                            encoder_embedding_dim,
                            hidden_dim,
                            n_layers,
                            encoder_dropout,
                        )
        self.decoder = Decoder(
                            output_dim,
                            decoder_embedding_dim,
                            hidden_dim,
                            n_layers,
                            decoder_dropout,
                        )
        self.model = Seq2Seq(self.encoder, self.decoder, device)
        self.model.apply(init_weights)
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    #     self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    #     self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        return self.model(state, teacher_forcing_ratio)

    
# https://github.com/BY571/SAC_discrete/blob/main/train.py
class SACAgent:
    def __init__(self, state_dim, action_dim, max_size=1000, batch_size=128, load_file='./sim/saved_model/sac_model/'):
        self.learning_rate = 5e-4
        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = 256
        self.clip_grad_param = 1
        self.target_entropy = -action_dim  # -dim(A)

        self.buffer_mlen = max_size
        self.batch_size = batch_size    
        self.load_file = load_file
        self.replay_buffer = BasicBuffer(batch_size, max_size=max_size)

        self.total_steps = 0
        self.update_step = 0
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.learning_rate) 
                
        # Actor Network 

        self.actor_local = Actor(state_dim, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_dim, hidden_size, 2).to(self.device)
        self.critic2 = Critic(state_dim, hidden_size, 1).to(self.device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_dim, hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_dim, hidden_size).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.learning_rate) 


    def get_action(self, state, action_space, is_training=True, eps=0.20, l=None):
        # if np.random.random() < eps: # random policy
        #     return np.random.choice(action_space[:l]), None
        
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            action = self.actor_local.get_det_action(state, l)

        return int(action), None
    
    def calc_policy_loss(self, states, alpha):
        log_action_pis = torch.tensor(0.0).to(self.device)
        actor_losses = torch.zeros(states.shape[0]).to(self.device)
        for i, state in enumerate(states):
            _, action_probs, log_pis = self.actor_local.evaluate(state)

            q1 = self.critic1(state)   
            q2 = self.critic2(state)
            min_Q = torch.min(q1,q2)
            actor_loss = (action_probs * (alpha * log_pis - min_Q )).sum(1).mean()
            log_action_pi = torch.sum(log_pis * action_probs, dim=1)
            log_action_pis += log_action_pi
            actor_losses[i] = actor_loss

        return actor_losses.mean(), log_action_pis
    
    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
    def update(self):
        batch = self.replay_buffer.sample()
        states, actions, rewards, next_states, dones, _, _ = batch

        states = torch.from_numpy(np.stack([state for state in states])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([action for action in actions])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([reward for reward in rewards])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([next_state for next_state in next_states])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([done for done in dones]).astype(np.uint8)).float().to(self.device)

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha.to(self.device))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.target_entropy = -actions.shape[1]
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)    # todolist
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 

        # Compute critic loss
        q1 = self.critic1(states).gather(1, actions.long())  # todolist: check if this is correct
        q2 = self.critic2(states).gather(1, actions.long())
        
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        self.update_step += 1

    def save(self, model_type="model"):  # model_type: best_model, best_reward_model, model
        os.makedirs(self.load_file, exist_ok=True)
        torch.save(self.actor_local.state_dict(), self.load_file+'{}_actor_local.pth'.format(model_type))


    def load(self, model_type="model"):  # model_type: best_model, best_reward_model, model
        if os.path.exists(self.load_file):
            self.actor_local.load_state_dict(torch.load(self.load_file+'{}_actor_local.pth'.format(model_type)))
            print("load model from {}".format(self.load_file+'{}_actor_local.pth'.format(model_type)))
        else:
            print("Model not found.")