import torch
import gym
from model import PPO


def test_render(agent, env, episodes=5):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        print(f"回合 {episode + 1}: 总奖励 = {total_reward}")
    env.close()


actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# 加载模型后测试
env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)
agent.actor.load_state_dict(torch.load('actor_model.pth'))
agent.actor.eval()
test_render(agent, env)
