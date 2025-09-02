import numpy as np
import gym
import time

env = gym.make('CartPole-v0')
state = env.reset()
done = False
action_desc = {0: "向左推", 1: "向右推"}
total_reward = 0.0
reward_list = []
gamma = 0.99

while not done:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, done, info = env.step(action)
    print(
        f"采取动作：{action_desc.get(action, '未知动作')}，获得奖励：{reward}，转移到下一个状态：{next_state}")
    reward_list.append(reward)
    time.sleep(0.1)

# 这里要注意回报是如何逆序计算的
for r in reward_list[::-1]:
    total_reward = r + gamma * total_reward

print('回报（收益）：', total_reward)
env.close()
