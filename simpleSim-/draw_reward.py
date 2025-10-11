import os
import matplotlib.pyplot as plt
import random

# log_path = ['log/SAC_FS.log', 'log/DQN_FS.log', 'log/PPO_FS.log', 'log/DQN_bys_pref_FS.log']
log_path = ['log/SAC_FS.log', 'log/DQN_FS.log', 'log/PPO_FS.log', 'log/DQN_bys_pref_FS.log']
log_name = ['SAC', 'DQN', 'PPO', 'DprefDQN(Ours)']
log_color = ['y', 'g', 'b', 'r']

save_path = 'saved_pic/'
os.makedirs(save_path, exist_ok=True)

all_rewards = []
for i, path in enumerate(log_path):
    each_rewards = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ', avg_reward of this episode ' in line:
                each_rewards.append(round(float(line.split(': ')[-1]), 3))
    all_rewards.append(each_rewards)


episode_num = 20
# max_len = max([len(rewards) for rewards in all_rewards])
max_len = episode_num
x = range(max_len)
for i in range(len(all_rewards)):
    rewards = all_rewards[i]
    if len(rewards) > episode_num:
        rewards = rewards[-episode_num:]
    # if len(rewards) < max_len:
    #     rewards += [rewards[-1]] * (max_len - len(rewards))
        
    # 打印绘制曲线的数组
    print(f"{log_name[i]} rewards: {rewards}")
    plt.plot(x, rewards, color=log_color[i], label=log_name[i])
plt.title('Rewards Comparison')
plt.xlabel('Episode')
plt.ylabel('Avg_reward')
plt.xticks(range(0, max_len+1, 5))
plt.legend()
plt.savefig(save_path+'reward.png')