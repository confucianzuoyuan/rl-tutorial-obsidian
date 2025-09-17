
```ad-note
近端策略优化（Proximal Policy Optimization，PPO），其性能与最先进的方法相当或更好，同时代码实现和优化起来更简单。
```

我们还是使用倒立摆环境来测试PPO算法。我们实现带裁剪目标的PPO算法。

首先定义策略的网络结构和价值函数的网络结构

## 网络结构的定义

```python
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class PolicyNet(torch.nn.Module):
	'''策略网络'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        # 对于倒立摆：state_dim = 4, hidden_dim = 128
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 对于倒立摆：action_dim = 2
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
	'''价值网络'''
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        # state_dim = 4
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 输出是一个价值，是标量
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

然后我们来实现带裁剪目标的PPO算法

## PPO训练器

```python
class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(
        self,
        state_dim, # 状态的维度 = 4
        hidden_dim, # 128
        action_dim, # 2个动作
        actor_lr, # 策略（演员）的学习率
        critic_lr, # 价值网络（评论家）的学习率
        lmbda, # λ，广义优势估计的超参数
        epochs, # 训练轮数
        eps, # ϵ，裁剪范围超参数(1-ϵ,1+ϵ)
        gamma, # γ折扣因子
        device # gpu or cpu
    ):
        # 演员是策略网络
        self.actor = PolicyNet(
            state_dim,
            hidden_dim,
            action_dim
        ).to(device)
        # 评论家是价值网络
        self.critic = ValueNet(
            state_dim,
            hidden_dim
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )
        self.gamma = gamma # 折扣因子
        self.lmbda = lmbda # GAE的超参数
        self.epochs = epochs
        self.eps = eps  # ϵ，截断范围
        self.device = device

	# 采取动作
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

	# 更新策略，每次智能体更新策略，都需要训练epochs=10轮
	# 更新一次智能体，需要训练10轮的策略网络和价值网络
    def update(self, transition_dict):
        # s_t 当前的状态，每一个时刻的 **当时的状态** 的数组
        states = torch.tensor(
            transition_dict['states'],
            dtype=torch.float
        ).to(self.device)
        # a_t 采取的动作，每一个时刻的采取的动作的数组
        actions = torch.tensor(
            transition_dict['actions']
        ).view(-1, 1).to(self.device)
        # r_t：t时刻获得的即时奖励组成的数组
        rewards = torch.tensor(
            transition_dict['rewards'],
            dtype=torch.float
        ).view(-1, 1).to(self.device)
        # s_{t+1} 每个时刻的 **下一个状态** 组成的数组
        next_states = torch.tensor(
            transition_dict['next_states'],
            dtype=torch.float
        ).to(self.device)
        # 是否结束标志位组成的数组
        dones = torch.tensor(
            transition_dict['dones'],
            dtype=torch.float
        ).view(-1, 1).to(self.device)
        # td_target = R_t + γV(S_{t+1})
        # 计算出每个时刻的单步TD目标组成的数组
        td_target = rewards + \
            self.gamma * self.critic(next_states) * (1 - dones)
        # δ = R_t + γV(S_{t+1}) - V(S_t)
        # 计算出每个时刻的单步TD误差组成的数组
        td_delta = td_target - self.critic(states)
        # A^{π_θ_old}_t
        # `compute_advantage` 计算广义优势估计
        # 每个时刻的GAE组成的数组[GAE_0, GAE_1, ..., GAE_n]
        advantage = compute_advantage(
            self.gamma,
            self.lmbda,
            td_delta.cpu()
        ).to(self.device)
        
        # 旧策略采取动作的 **对数概率** log{π_θ_old(a_t|s_t)}
        # 冻结（保存）一份 **旧策略** 的采取动作的对数概率的数组
        old_log_probs = torch.log(
            # todo: gather讲解
            # 取出每个时刻的采取的那个具体动作的概率
            # dim = 1, index = actions
            # gather一般用在使用label标签从softmax后的数组中抽取对应的概率
            self.actor(states).gather(1, actions)
        ).detach()
        # 训练epochs=10轮
        for _ in range(self.epochs):
            # 新策略采取动作的 **对数概率** log{π_θ_new(a_t|s_t)}
            # 第1轮训练的时候，新旧策略模型是相同模型
            # 新的策略用旧策略产生的轨迹的状态输出动作的概率
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 比率计算公式
            ratio = torch.exp(log_probs - old_log_probs)
            # surr --> surrogate替代
            # p_t(θ)*优势，公式中逗号的左边项
            surr1 = ratio * advantage
            # 公式中逗号的右边项
            surr2 = torch.clamp(
                ratio, # p_t(θ)
                1 - self.eps, # 1-ϵ = 0.8
                1 + self.eps # 1+ϵ = 1.2
            ) * advantage
            # PPO损失函数
            # 公式中min的实现 `-torch.min(surr1, surr2)`
            # 策略的损失
            # 每个时刻新旧策略采取动作a_t的比率的 **均值**
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失
            # 每一步的TD误差的平均值
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
```

```ad-tip
title: 比率的计算

$$
\frac{A}{B} = e^{\log{A}-\log{B}} = \frac{e^{\log A}}{e^{\log B}}
$$
```

算法中计算广义优势估计的代码 `compute_advantage` 如下

```python
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
```

![[ppo-algo-array.excalidraw|1000]]

接下来，在倒立摆环境中训练 PPO 算法。

## 训练循环

```python
actor_lr = 1e-3 # 策略网络的学习率
critic_lr = 1e-2 # 价值网络的学习率
num_episodes = 500 # 玩500个回合的游戏
hidden_dim = 128
gamma = 0.98 # 折扣因子
lmbda = 0.95 # 广义优势估计用到的超参数λ
epochs = 10 # 智能体每次更新策略需要训练10轮
eps = 0.2 # ϵ = 0.2
device = torch.device("cuda") \
    if torch.cuda.is_available() \
    else torch.device("cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
# state_dim = 4
state_dim = env.observation_space.shape[0]
# action_dim = 2
action_dim = env.action_space.n
# 初始化智能体
agent = PPO(
    state_dim,
    hidden_dim,
    action_dim,
    actor_lr,
    critic_lr,
    lmbda,
    epochs,
    eps,
    gamma,
    device
)
# 训练策略
return_list = train_on_policy_agent(env, agent, num_episodes)
```

其中 `train_on_policy_agent` 如下

```python
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(
            total=int(num_episodes/10),
            desc='Iteration %d' % i
        ) as pbar:
            # 循环 500 / 10 = 50
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {
	                'states': [],
	                'actions': [],
	                'next_states': [],
	                'rewards': [],
	                'dones': []
	            }
                state = env.reset()
                done = False
                # 采样一条轨迹，记录了每个动作的相关信息
                # 注意：这条轨迹会被反复使用10次，样本利用率比较高
                while not done:
                    # 每执行一次动作，就将相关信息添加到对应的数组
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                # 更新策略 ---> 冻结一份旧策略的概率，然后更新10次策略网络
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (
                            num_episodes/10 * i + i_episode+1
                        ),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list
```

可视化一下训练结果

```python
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('CartPole-v0')
plt.show()
```

![[使用PPO玩倒立摆的收益变化图.png]]

```ad-danger
PPO 算法是 OpenAI 使用的默认算法。
```

## 反向传播算法

![[backprop.excalidraw|1000]]