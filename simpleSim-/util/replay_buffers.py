import random

class BasicBuffer:
    def __init__(self, batch_size=128, max_size=1000):
        self.buffer = []
        self.max_size = max_size
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done, a_logprob=None, dw=None):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append([state, action, reward, next_state, done, a_logprob, dw])

    def reset(self):
        self.buffer = []

    def get_rewards(self):
        rewards = []
        for experience in self.buffer:
            rewards.append(experience[2])
        return rewards
    
    def reset_rewards(self, new_rewards):
        for i in range(len(self.buffer)):
            self.buffer[i][2] = new_rewards[i]

    def sample(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        a_logprob_batch = []
        dw_batch = []

        # print(f"buffer len:{len(self.buffer)}")
        # print(f"batch size:{self.batch_size}")
        batch = random.sample(self.buffer, self.batch_size)

        for experience in batch:
            state, action, reward, next_state, done, a_logprob, dw = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            a_logprob_batch.append(a_logprob)
            dw_batch.append(dw)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch, a_logprob_batch, dw_batch)
    
    def __len__(self):
        return len(self.buffer)
