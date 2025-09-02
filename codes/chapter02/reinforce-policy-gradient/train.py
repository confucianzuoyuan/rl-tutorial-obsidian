import matplotlib.pyplot as plt
import gym
import torch

from model import Agent


env = gym.make('CartPole-v0')
agent = Agent()
reward_history = []

for episode in range(3000):
    # 重置倒立摆环境的状态
    state = env.reset()
    done = False
    total_reward = 0

    # 采样一条轨迹
    while not done:
        # 根据环境的当前状态决策要采取什么动作
        action, prob = agent.get_action(state)
        # 采取动作，环境会返回环境的下一个状态，奖励和是否结束游戏
        next_state, reward, done, _ = env.step(action)

        # 将奖励和动作的概率保存下来
        agent.add(reward, prob)
        # 环境转移到下一个状态
        state = next_state
        # 累积奖励
        total_reward += reward

    # 使用反向传播算法更新策略神经网络
    agent.update()
    reward_history.append(total_reward)

    if episode % 100 == 0:
        print("回合:{}, 总奖励:{:.1f}".format(episode, total_reward))

torch.save(agent.pi.state_dict(), 'policy_model.pth')
print("模型已保存")

# 训练结束后绘制奖励变化图
plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'  # 显示中文
plt.plot(reward_history)
plt.xlabel('回合')
plt.ylabel('总奖励')
plt.title('每回合奖励')
plt.grid(True)
plt.show()
