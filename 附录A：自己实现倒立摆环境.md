
```python
import math
import numpy as np
import matplotlib.pyplot as plt


class CartPoleEnv:
    def __init__(self):
        # 环境参数（与OpenAI Gym的CartPole-v0类似）
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # 杆长的一半
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # 时间步长（秒）
        self.max_steps = 200  # 最大步数限制
        self.current_step = 0

        # 终止条件
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  # 12度转弧度
        self.x_threshold = 2.4

        self.state = None  # 存储状态变量
        self.steps_beyond_done = None

        # 画图参数
        self.fig = None
        self.ax = None

    def reset(self):
        # 状态：x, x_dot, theta, theta_dot
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.current_step = 0
        return np.array(self.state)

    def step(self, action):
        """
        action: 0 或 1，分别表示向左或向右施加力
        返回: next_state, reward, done, info
        """
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # 物理方程，参考OpenAI Gym源码
        temp = (force + self.polemass_length * theta_dot **
                2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0/3.0 - self.masspole *
                    costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # 更新状态
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)
        
        self.current_step += 1

        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians \
            or self.current_step >= self.max_steps  # 步数限制

        done = bool(done)

        reward = 1.0 if not done else 0.0

        return np.array(self.state), reward, done, {}

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()
            plt.show()

        x, _, theta, _ = self.state

        self.ax.clear()

        # 画轨道
        self.ax.plot(
            [-self.x_threshold*1.2, self.x_threshold*1.2], [0, 0], 'k-')

        # 画小车
        cart_y = 0.1
        cart_width = 0.3
        cart_height = 0.2
        self.ax.add_patch(plt.Rectangle((x - cart_width/2, cart_y - cart_height/2),
                                        cart_width, cart_height, color='blue'))

        # 画杆
        pole_x = x + self.length * math.sin(theta)
        pole_y = cart_y + cart_height/2 + self.length * math.cos(theta)
        self.ax.plot([x, pole_x], [cart_y + cart_height /
                     2, pole_y], 'r-', linewidth=4)

        self.ax.set_xlim(-self.x_threshold*1.5, self.x_threshold*1.5)
        self.ax.set_ylim(-0.5, 1.0)
        self.ax.set_aspect('equal')
        self.ax.set_title('CartPole-v0')
        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None


# 测试环境
if __name__ == "__main__":
    env = CartPoleEnv()
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = np.random.choice([0, 1])  # 随机动作
        state, reward, done, _ = env.step(action)
        print('state: ', state)
        print('reward: ', reward)
    env.close()
```

