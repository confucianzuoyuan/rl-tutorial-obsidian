import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98  # 折扣因子
        self.lr = 0.0002  # 学习率
        self.action_size = 2  # 动作空间大小，共两个动作：向左推和向右推

        self.memory = []
        self.pi = Policy(self.action_size)  # 初始化策略神经网络
        # 使用 Adam 优化器
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        # 将状态转换成torch.tensor类型
        state = torch.tensor(state[np.newaxis, :])
        # 将状态输入策略神经网络，输出为两个动作的概率分布
        probs = self.pi(state)
        # 取出概率分布
        probs = probs[0]
        # 下面两行根据动作的分布采样出一个动作
        m = Categorical(probs)
        action = m.sample().item()

        # 返回动作和动作的概率
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
        for reward, prob in self.memory:
            loss += - prob.log() * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []  # 重置内存
