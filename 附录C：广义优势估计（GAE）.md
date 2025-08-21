
$$
\begin{aligned}
V_\pi(S_t) &= \mathbb{E}_\pi[G_t|S_t] \\
&= \mathbb{E}_\pi\left[\sum_{k=0}^\infty\gamma^kR_{t+k}\mid S_t\right] \\
&= \mathbb{E}_\pi\left[R_t+\gamma\sum_{k=0}^\infty\gamma^kR_{t+k+1}\mid S_t\right] \\
&= \mathbb{E}_\pi[R_t+\gamma G_{t+1}|S_t] \\
&= \mathbb{E}_\pi[R_t|S_t]+\gamma \mathbb{E}_\pi[G_{t+1}|S_t] \\
&= \mathbb{E}_\pi[R_t|S_t]+\gamma \mathbb{E}_\pi[\mathbb{E}_\pi [G_{t+1}|S_{t+1}]|S_t] \\
&= \mathbb{E}_\pi[R_t|S_t]+\gamma \mathbb{E}_\pi[V_\pi(S_{t+1})|S_t] \\
&= \mathbb{E}_\pi[R_t+\gamma V_\pi(S_{t+1})|S_t] \\
\end{aligned}
$$

```ad-note
全期望公式：$\mathbb{E}[X]=\mathbb{E}[\mathbb{E}[X|Y]]$
条件期望的迭代公式：$\mathbb{E}[X|Y]=\mathbb{E}[\mathbb{E}[X|Z]|Y]$
```

我们如何估计或者计算优势函数 $A$ 呢？目前比较常用的一种方法为 **广义优势估计**（Generalized Advantage Estimation，GAE），接下来我们简单介绍一下 $\text{GAE}$ 的做法。首先，用

$$
\delta_t=R_t+\gamma V(s_{t+1})-V(s_t)
$$

表示时序差分误差（TD误差）。其中 $V$ 是一个价值函数神经网络。于是，根据多步时序差分的思想，有：

$$
\begin{aligned}
A_t^{(1)} &= \delta_t &&= -V(s_t)+R_t+\gamma V(s_{t+1}) \\
A_t^{(2)} &= \delta_t+\gamma\delta_{t+1} &&= -V(s_t)+R_t+\gamma R_{t+1}+\gamma^2 V(s_{t+2}) \\
A_t^{(3)} &= \delta_t+\gamma\delta_{t+1}+\gamma^2 \delta_{t+2} &&= -V(s_t)+R_t+\gamma R_{t+1}+ \gamma^2R_{t+2} +\gamma^3 V(s_{t+3}) \\
&\quad\vdots &&\quad\quad\vdots \\
A_t^{(k)} &= \sum_{l=0}^{k-1}\gamma^l\delta_{t+l} &&= -V(s_t)+R_t+\gamma R_{t+1}+\cdots+\gamma^{k-1}R_{t+k-1}+\gamma^kV(s_{t+k})
\end{aligned}
$$

然后，GAE 将这些不同步数的优势估计进行指数加权平均：

$$
\begin{aligned}
A_t^{GAE} &= (1-\lambda)(A_t^{(1)}+\lambda A_t^{(2)}+\lambda^2A_t^{(3)}+\cdots) \\
&= (1-\lambda)(\delta_t+\lambda(\delta_t+\gamma\delta_{t+1})+\lambda^2(\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2})+\cdots) \\
&= (1-\lambda)(\delta_t(1+\lambda+\lambda^2+\cdots)+\gamma\delta_{t+1}(\lambda+\lambda^2+\lambda^3+\cdots)+\gamma^2\delta_{t+2}(\lambda^2+\lambda^3+\lambda^4+\cdots)+\cdots) \\
&= (1-\lambda)\left({\delta_t\frac{1}{1-\lambda}+\gamma\delta_{t+1}\frac{\lambda}{1-\lambda}+\gamma^2\delta_{t+2}\frac{\lambda^2}{1-\lambda}+\cdots}\right) \\
&= \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
\end{aligned}
$$

其中，$\lambda\in [0,1]$ 是在 GAE 中额外引入的一个超参数。当 $\lambda=0$ 时，$A_t^{GAE}=\delta_t=R_t+\gamma V(s_{t+1})-V(s_t)$ ，也就是仅仅只看一步差分得到的优势；当 $\lambda=1$ 时，$A_t^{GAE}=\sum_{l=0}^{\infty}\gamma^l\delta_{t+l}=\sum_{l=0}^{\infty}\gamma^lR_{t+l}-V(s_t)$ ，则是看每一步差分得到的优势的完全平均值。

有上面的式子，我们还可以推导出一个递推公式

```ad-note
$$
A_t = \delta_t + \gamma\lambda A_{t+1}
$$
```

下面是计算 GAE 的代码，给定 $\gamma$ 和 $\lambda$ 以及每个时间步的 $\delta_t$ 之后，我们可以根据公式直接进行优势估计。

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

