"""
PPO implementation inspired by the StableBaselines3 implementation.
To reuse trained models, you can make use of the save and load function
To adapt policy and value network structure, specify the policy and value layer and activation parameter
in your train config or change the constants in this file
"""
import os
import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import pickle
from typing import Tuple, Any, List

# constants
POLICY_LAYER: List[int] = [256, 256]
POLICY_ACTIVATION: str = 'ReLU'
VALUE_LAYER: List[int] = [256, 256]
VALUE_ACTIVATION: str = 'ReLU'


class RolloutBuffer:
    """
    Handles episode data collection and batch generation

    :param buffer_size: Buffer size
    :param batch_size: Size for batches to be generated

    """
    def __init__(self, buffer_size: int, batch_size: int):

        self.observations = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.advantages = None
        self.returns = None

        if buffer_size % batch_size != 0:
            raise TypeError("rollout_steps has to be a multiple of batch_size")
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.reset()

    def generate_batches(self) -> Tuple:
        """
        Generates batches from the stored data

        :return: batches: Lists with all indices from the rollout_data, shuffled and sampled in lists with batch_size
            e.g. [[0,34,1,768,...(len: batch size)], [], ...(len: len(rollout_data) / batch size)]

        """
        # create random index list and split into arrays with batch size
        indices = np.random.permutation(self.buffer_size)
        num_batches = int(self.buffer_size / self.batch_size)
        batches = indices.reshape((num_batches, self.batch_size))

        return np.array(self.observations), np.array(self.actions), np.array(self.probs), batches

    def compute_advantages_and_returns(self, last_value, gamma, gae_lambda) -> None:
        """
        Computes advantage values and returns for all stored episodes.

        :param last_value: Value from the next step to calculate the advantage for the last episode in the buffer
        :param gamma: Discount factor for the advantage calculation
        :param gae_lambda: Smoothing parameter for the advantage calculation

        :return: None

        """
        # advantage: advantage from the actual returned rewards over the baseline value from step t onwards
        last_advantage = 0
        for step in reversed(range(self.buffer_size)):
            # use the predicted reward for the advantage computation of the last step of the buffer
            if step == self.buffer_size - 1:
                # if a step is the last one of the episode (done = 1) -> not_done = 0 => the advantage
                # doesn't contain values outside the own episode
                not_done = 1.0 - self.dones[step]
                next_values = last_value
            else:
                not_done = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * not_done - self.values[step]
            last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
            self.advantages[step] = last_advantage

        # compute returns = discounted rewards, advantages = discounted rewards - values
        # Necessary to update the value network
        self.returns = self.values + self.advantages

    def push(self, observation: np.ndarray, action: int, prob: float, value: float,
                     reward: Any, done: bool) -> None:
        """
        Appends all data from the recent step

        :param observation: Observation at the beginning of the step
        :param action: Index of the selected action
        :param prob: Probability of the selected action (output from the policy_net)
        :param value: Baseline value that the value_net estimated from this step onwards according to the
        :param observation: Output from the value_net
        :param reward: Reward the env returned in this step
        :param done: True if the episode ended in this step

        :return: None

        """
        self.observations.append(observation)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def reset(self) -> None:
        """
        Resets all buffer lists

        :return: None

        """
        self.observations = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)


class PolicyNetwork(nn.Module):
    """
    Policy Network for the agent

    :param input_dim: Observation size to determine input dimension
    :param n_actions: Number of action to determine output size
    :param learning_rate: Learning rate for the network
    :param hidden_layers: List of hidden layer sizes (int)
    :param activation: String naming activation function for hidden layers

    """
    def __init__(self, input_dim: int, n_actions: int, learning_rate: float, hidden_layers: List[int], activation: str):

        super(PolicyNetwork, self).__init__()

        net_structure = []
        # get activation class according to string
        activation = getattr(nn, activation)()

        # create first hidden layer in accordance with the input dim and the first hidden dim
        net_structure.extend([nn.Linear(input_dim, hidden_layers[0]), activation])

        # create the other hidden layers
        for i, layer_dim in enumerate(hidden_layers):
            if not i+1 == len(hidden_layers):
                net_structure.extend([nn.Linear(layer_dim, hidden_layers[i+1]), activation])
            else:
                # create output layer
                net_structure.extend([nn.Linear(layer_dim, n_actions), nn.Softmax(dim=-1)])

        self.policy_net = nn.Sequential(*net_structure)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        """forward function"""
        observation.to(self.device)
        logits = self.policy_net(observation)

        dist = Categorical(logits=logits)
        
        return dist


class ValueNetwork(nn.Module):
    """
    Value Network for the agent

    :param input_dim: Observation size to determine input dimension
    :param learning_rate: Learning rate for the network
    :param hidden_layers: List of hidden layer sizes (int)
    :param activation: String naming activation function for hidden layers

    """
    def __init__(self, input_dim: int, learning_rate: float, hidden_layers: List[int], activation: str):
        super(ValueNetwork, self).__init__()

        net_structure = []
        # get activation class according to string
        activation = getattr(nn, activation)()

        # create first hidden layer in accordance with the input dim and the first hidden dim
        net_structure.extend([nn.Linear(input_dim, hidden_layers[0]), activation])

        # create the other hidden layers
        for i, layer_dim in enumerate(hidden_layers):
            if not i + 1 == len(hidden_layers):
                net_structure.extend([nn.Linear(layer_dim, hidden_layers[i + 1]), activation])
            else:
                # create output layer
                net_structure.append(nn.Linear(layer_dim, 1))

        self.value_net = nn.Sequential(*net_structure)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        """forward function"""
        value = self.value_net(observation)

        return value


class PPOAgent:
    """PPO Agent class"""
    def __init__(self, state_dim, action_dim, batch_size=128, load_file='./sim/saved_model/ppo_model/'):
        """
        | gamma: Discount factor for the advantage calculation
        | learning_rate: Learning rate for both, policy_net and value_net
        | gae_lambda: Smoothing parameter for the advantage calculation
        | clip_range: Limitation for the ratio between old and new policy
        | batch_size: Size of batches which were sampled from the buffer and fed into the nets during training
        | n_epochs: Number of repetitions for each training iteration
        | rollout_steps: Step interval within the update is performed. Has to be a multiple of batch_size
          执行更新时的步骤间隔。必须是 batch_size 的倍数
        """

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.n_epochs = 5
        self.rollout_steps = 1024
        self.ent_coef = 0.0
        self.n_updates = 0
        self.learning_rate = 0.002
        self.batch_size = batch_size
        self.seed = 0
        self.load_file = load_file

        # torch seed setting
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            T.manual_seed(self.seed)

        # create networks and buffer  256还是64呢，源玛和batchsize一样是256
        self.policy_net = PolicyNetwork(state_dim, action_dim, self.learning_rate,
                                        [ 256, 256 ],
                                        'ReLU')
        self.value_net = ValueNetwork(state_dim, self.learning_rate,
                                      [ 256, 256 ],
                                      'ReLU')
        self.replay_buffer = RolloutBuffer(self.rollout_steps, self.batch_size)

    def load(self, ifbest=True):
        if ifbest and os.path.exists(self.load_file+"_value_best_model.pth") and os.path.exists(self.load_file+"_policy_best_model.pth"):
            # self.model.load_state_dict(torch.load(self.load_file+"best_model.pth"))
            self.value_net = T.load(self.load_file+"_value_best_model.pth")
            self.policy_net = T.load(self.load_file+"_policy_best_model.pth")
            print("load model from {}".format(self.load_file+"best_model"))
        elif os.path.exists(self.load_file+"_value_model.pth") and os.path.exists(self.load_file+"_policy_model.pth"):
            # self.model.load_state_dict(torch.load(self.load_file+"model.pth"))
            self.value_net = T.load(self.load_file+"_value_model.pth")
            self.policy_net = T.load(self.load_file+"_policy_model.pth")
            print("load model from {}".format(self.load_file+"model"))
        else:
            print("Model not found.")

    def save(self, ifbest=False):
        os.makedirs(self.load_file, exist_ok=True)
        if ifbest:
            T.save(self.value_net, self.load_file+"_value_best_model.pth")
            T.save(self.policy_net, self.load_file+"_policy_best_model.pth")
            # print("save model to {}".format(self.load_file+"best_model"))
        else:
            T.save(self.value_net, self.load_file+"_value_model.pth")
            T.save(self.policy_net, self.load_file+"_policy_model.pth")
            # print("save model to {}".format(self.load_file+"model"))

    def forward(self, observation: np.ndarray, **kwargs) -> Tuple:
        """
        Predicts an action according to the current policy based on the observation
        and the value for the next state

        :param observation: Current observation of teh environment
        :param kwargs: Used to accept but ignore passing actions masks from the environment.

        :return: Predicted action, probability for this action, and predicted value for the next state

        """

        observation = T.tensor(observation, dtype=T.float).to(self.policy_net.device)

        dist = self.policy_net(observation)
        value = self.value_net(observation)
        action = dist.sample()

        prob = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, prob, value

    def get_action(self, observation, action_space, is_training, eps, deterministic=True):
        """
         Action prediction for testing

        :param observation: Current observation of teh environment
        :param deterministic: Set True, to force a deterministic prediction
        :param state: The last states (used in rnn policies)
        :param kwargs: Used to accept but ignore passing actions masks from the environment.

        :return: Predicted action and next state (used in rnn policies)

        """
        action, prob, val = self.forward(observation)

        return action, prob, val

    def train(self) -> None:
        """
        Trains policy and value

        :return: None

        """
        # switch to train mode
        self.policy_net.train(True)
        self.value_net.train(True)

        policy_losses, value_losses, entropy_losses, total_losses = [], [], [], []

        for _ in range(self.n_epochs):

            # get data from buffer and random batches(index lists) to iterate over
            # e.g. obs[batch] returns the observations for all indices in batch
            obs_arr, action_arr, old_prob_arr, batches = self.replay_buffer.generate_batches()

            # get advantage and return values from buffer
            advantages = T.tensor(self.replay_buffer.advantages).to(self.policy_net.device)
            returns = T.tensor(self.replay_buffer.returns).to(self.value_net.device)

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for batch in batches:
                observations = T.tensor(obs_arr[batch], dtype=T.float).to(self.policy_net.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.policy_net.device)
                actions = T.tensor(action_arr[batch]).to(self.policy_net.device)

                dist = self.policy_net(observations)
                values = self.value_net(observations)
                values = T.squeeze(values)

                # ratio between old and new policy (probs of selected actions)
                # Should be one at the first batch of every train iteration
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                # policy clip
                policy_loss_1 = prob_ratio * advantages[batch]
                policy_loss_2 = T.clamp(prob_ratio, 1-self.clip_range, 1+self.clip_range) * advantages[batch]
                # we want to maximize the reward, but running gradient descent -> negate the loss here
                policy_loss = -T.min(policy_loss_1, policy_loss_2).mean()

                value_loss = (returns[batch]-values)**2
                value_loss = value_loss.mean()

                # entropy loss
                entropy_loss = -T.mean(dist.entropy())
                entropy_losses.append(entropy_loss.item())

                total_loss = policy_loss + 0.5*value_loss + self.ent_coef*entropy_loss
                self.policy_net.optimizer.zero_grad()
                self.value_net.optimizer.zero_grad()
                total_loss.backward()
                self.policy_net.optimizer.step()
                self.value_net.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                total_losses.append(total_loss.item())

        self.n_updates += self.n_epochs

        print(f'losses: {np.mean(total_losses)}')


    def update(self, new_obs) -> None:
        with T.no_grad():
            _, _, val = self.forward(new_obs)
        self.replay_buffer.compute_advantages_and_returns(val, self.gamma, self.gae_lambda)

        # train networks
        self.train()
        # switch back to normal mode
        self.policy_net.train(False)
        self.value_net.train(False)

        # reset buffer to continue collecting rollouts
        self.replay_buffer.reset()