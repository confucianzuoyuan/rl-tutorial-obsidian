
策略梯度方法中，除了传统的算法以外，还有信任区域策略优化（trust region policy optimization，TRPO）算法和近端策略优化（proximal policy optimization，PPO）算法。TRPO和PPO的基本想法相同，都是将策略更新控制在确信的范围内，以保证学习的稳定性。PPO是TPRO的改进，具有TRPO的性能单调递增、训练稳定、样本使用效率高等优点的同时，也具有实现简单、训练效率高的优点。

传统的策略梯度算法存在训练不稳定问题。策略更新时，幅度过大有时会导致策略的性能急剧下降，后续无法学到更优的策略。就好像在登山的途中，原本沿着一条既定的路径向上攀登，却因为在某个时刻迈出的步子过大，结果偏离正确路径，跌落至悬崖之下。

TRPO以新旧策略之间的KL散度定义信任区域,将策略更新限制在信任区间之内。基于旧策略进行数据采样，优势函数计算，对新策略进行更新。理论上保证在一定条件下策略性能的单调提升。

PPO与TRPO有相同的框架，不同的是，使用概率比值来衡量新旧策略之间的差异，通过设定概率比值的取值范围限制策略更新的幅度。经验上，也能取得与TRPO同等的效果。

Schulman等于2015年提出了TRPO算法，接着Schulman等于2017年提出了PPO算法。TRPO奠定了理论基础，PPO提供出了简单实现。目前PPO是强化学习中最常用的算法之一。已经应用于AI游戏、机器人控制、大语言模型微调等诸多问题。特别是ChatGPT等大语言模型使用PPO算法进行对齐。

本章讲述TRPO和PPO算法，特别是后者。第1节讲述TRPO算法，第2节讲述PPO算法，第3节介绍PPO在大语言模型ChatGPT上的应用。

## 1 TRPO算法

本节阐述TRPO算法要解决的问题，呈现算法的基本形式，给出算法的推导和理论支持。

### 1.1 背景和动机

基于策略的方法尝试直接学习最优策略，相比基于价值的方法，往往能更有效地达到学习的目的。演员-评论员使用优势函数评价策略的性能，在这些方法中效果更好。

演员-评论员算法尝试通过迭代的方法不断提升策略函数的性能。在迭代过程中，使用当前策略 $\pi_{\theta}(a|s)$ 进行数据采样，计算当前策略的优势函数，通过随机梯度上升法来改进当前策略。形式化为一个优化问题，目标是更新策略函数，使得当前策略的回报期望最大。根据策略梯度定理，目标函数的梯度函数写作

$$
\nabla J(\theta) = \mathbb{E}_{\rho_{\theta}(a|s)}\left[ \mathbb{E}_{\pi_{\theta}(a|s)} \left[ \nabla_{\theta} \log \pi_{\theta}(a|s)A_{\pi_{\theta}}(s,a) \right]  \right] \tag{1} 
$$

其中，$\pi_{\theta}(a|s)$ 是当前策略的策略函数，$\rho_{\theta}(s)$ 是基于当前策略和环境的状态 $s$ 的访问分布，$A_{\pi_{\theta}}(a|s)$ 是优势函数。策略更新时基于采样样本估计梯度函数值。

传统的策略梯度算法包括演员-评论员算法，都存在学习不稳定和采样效率低的问题。

演员-评论员在随机梯度上升迭代的每一步，梯度函数决定了策略函数改进的方向，步长决定了策略函数改进的幅度。理想情况，步长在每一步是可变的，而不是不变的，因为策略函数的改进幅度需要根据学习状况决定。与监督学习不同，强化学习的数据不是固定的，而是通过随机采样（与环境交互的经验）得到的，因此是动态变化的。不同的策略从环境得到的观测（状态）和奖励不同，也就是得到的数据不同。如果选择的策略不当，进入不理想的区域采样，那么就无法学到性能很高的策略。步长过大，就有进入不理想区域的风险。减小步长能降低这些风险，但迭代的步数会增加，学习的效率会降低。

采样效率不高是另一个问题。演员-评论员是在策略（on-policy）学习算法，使用随机梯度上升进行当前策略的改进，这时用当前策略采样得到数据，对数据进行一次遍历（one epoch），更新当前策略，得到下一步的策略。也就是说，目标策略和行为策略是一致的（是在策略学习）。在下一步，将数据丢弃，从下一步的策略继续进行学习。每一步采样得到的数据在学习中都没有被充分利用。此外，因为模型更新依赖于采样数据，采样和更新两个阶段无法并行进行，学习效率也不高。

### 1.2 基本形式

传统的策略梯度算法可能会因为策略更新幅度过大，无法学到最优策略。TRPO（trust region policy optimization）算法引入信任区域（trust region）的概念，在每一次策略更新中，用新旧策略之间的KL散度表示信任区域，将策略函数的更新幅度限制在信任区域内。

TRPO是策略梯度，特别是演员-评论员算法的改进，属于在策略学习算法。TRPO也道循强化学习的一般原理，由数据采样、策略评估、策略改进三步组成。在迭代过程中，前一步的策略 $\pi_{\theta(a|s)}$ 称为旧策略，当前一步的策略称为新策略 $\pi_{\theta'}(a|s)$ 。首先使用旧策略进行数据采样，然后计算旧策略的优势函数，接着在此基础上通过随机梯度上升对新策略进行改进。策略改进形式化为有约束的优化问题。目标是更新新策略的参数，使得新旧策略比值与旧策略优势函数之积关于旧策略的期望最大。约束条件由新旧策略之间的KL散度表示。有约束的优化问题写作

$$
\begin{aligned}
\max_{\theta'}[L(\theta';\theta)] = \max_{\theta'}\mathbb{E}_{\rho_{\theta}(s)}\left[ \mathbb{E}_{\pi_{\theta}(a|s)}\left[ \frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}A_{\pi_{\theta}}(s,a) \right]  \right] \\ \\
\forall s,\mathbb{E}_{\rho_{\theta}(s)}[\text{KL}(\pi_{\theta}(a|s)\vert \pi_{\theta'}(a|s))]\leq\delta
\end{aligned} \tag{2}
$$

其中，$\pi_{\theta}(a|s)$ 是旧策略的策略函数，$\pi_{\theta'}(a|s)$ 是新策略的策略函数，$\rho_{\theta}(s)$ 是基于旧策略和环境的状态 $s$ 的访问分布，$A_{\pi_{\theta}}(s,a)$ 是旧策略的优势函数，$\delta$ 是超参数。模型既可以是有限期MDP，也可以是无限期MDP。

目标函数的梯度函数是

$$
\nabla_{\theta'}L(\theta;\theta)=\mathbb{E}_{\rho_{\theta}(s)}\left[ \mathbb{E}_{\pi_{\theta}(a|s)}\left[ \frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}\nabla_{\theta'}\log{\pi_{\theta'}}(a|s)A_{\pi_{\theta}}(s,a) \right]  \right] \tag{3}
$$

新旧策略的策略函数的比值写作

$$
r_{\theta'}(s,a) = \frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)} \tag{4}
$$

比值 $r_{\theta'}(s,a)$ 表示新策略 $\pi_{\theta'}(s,a)$ 和旧策略 $\pi_{\theta}(s,a)$ 在状态 $s$ 下动作 $a$ 的概率的比值，取正值。如果这个比值大于 $1$ ，则新策略比旧策略在状态 $s$ 更倾向于采取动作 $a$ 。如果这个比值小于 $1$ ，则新策略比旧策略在状态 $s$ 更倾向于不采取动作 $a$ 。

优势函数 $A_{\pi_{\theta}}(s,a)$ 表示，基于旧策略与环境交互的经验，在状态 $s$ 下采取动作 $a$ 的价值变化，取实数值。如果 $A_{\pi_{\theta}}(s,a)$ 大于 $0$ ，则采取动作 $a$ 有价值的增益。如果优势值小于 $0$ ，则采取动作 $a$ 没有价值的增益。

当 $A_{\pi_{\theta}}(s,a)$ 大于 $0$ 时，改变新策略 $\pi_{\theta'}(a|s)$ 的参数，使比值 $r_{\theta'}(s,a)$ 大于 $1$ ，就能促使目标函数值提升，得到性能更高的策略。当 $A_{\pi_{\theta}}(s,a)$ 小于 $0$ 时，改变新策略 $\pi_{\theta'}(a|s)$ 的参数，比值 $r_{\theta'}(s,a)$ 小于 $1$ ，就能抑制目标函数值降低，得到性能更高的策略。

如后面描述，目标函数包含重要性采样，用于校正新旧策略之间的分布差异的影响。这使得算法能够基于旧策略的采样数据，针对新策略进行策略改进。事实上，优化问题(2)中的新旧策略的比值来自重要性采样。

新旧策略之间的KL散度决定信任区域。优化的过程中，作为约束条件，保证新策略与旧策略的KL散度小于 $\delta$ ，也就是说，更新旧策略得到的新策略一定在信任区间之内。注意信任区间是定义在策略的概率分布空间，而不是在策略的参数空间。

演员-评论员和TRPO算法都是通过迭代方式尝试不断改进策略函数。TRPO能保证每次迭代性能都能得到提高，而演员-评论员并不能保证。在形式上两个算法也有很多不同。演员-评论员在每一步对当前策略进行改进，而TRPO在每一步利用前一步策略对当前策略进行改进。演员-评论员的数据采样和优势函数都基于当前策略，TRPO的数据采样和优势函数计算基于前一步策略。比较式(1)和式(3)可以看出，两个算法的梯度函数也不相同。但是，当 $\theta'=\theta$ 时，TRPO退化成为演员-评论员，梯度函数成为

$$
\nabla_{\theta}[L(\theta;\theta)] = \mathbb{E}_{\rho_{\theta}(a|s)}[\mathbb{E}_{\pi_{\theta}(a|s)}[\nabla_{\theta}{ \log \pi_{\theta}}(a|s)A_{\pi_{\theta}}(s,a)]]
$$

约束条件恒真。

### 1.3 算法和理论推导

下面讲述TRPO的算法推导、直观解释和理论保证。假设模型是无限期MDP。

策略梯度算法的目标是最大化价值函数，也就是回报的期望。价值函数表示策略的性能高低。假设有两个策略，分别是新策略 $\pi_{\theta'}(\tau)$ 和旧策略 $\pi_{\theta}(\tau)$ ，那么它们的价值函数分别定义为

$$
J(\theta') = \mathbb{E}_{\pi_{\theta'}(\tau)}\left[ \sum_{t=0}^\infty\gamma^tR(S_{t, A_{t}}) \right] = \mathbb{E}_{P(S_{0})}\left[ V_{\pi_{\theta'}}(S_{0}) \right] \tag{5}  
$$

$$
J(\theta) = \mathbb{E}_{\pi_{\theta}(\tau)}\left[ \sum_{t=0}^\infty\gamma^tR(S_{t, A_{t}}) \right] = \mathbb{E}_{P(S_{0})}\left[ V_{\pi_{\theta}}(S_{0}) \right] \tag{6}
$$

如果

$$
J(\theta')-J(\theta)\geq 0
$$

成立，那么新策略就比旧策略的性能更高。事实上，TRPO算法保证在迭代的每一步这个条件都能得到满足，即性能是单调递增的。也就是说，可以将TRPO看作一种策略改进的方法。首先，关于 $J(\theta')-J(\theta)$ 有以下引理成立。

**引理 1**

$$
J(\theta')-J(\theta) = \mathbb{E}_{\pi_{\theta'}(\tau)}\left[ \sum^\infty_{t=0}\gamma^tA_{\pi_{\theta}}(S_{t},A_{t}) \right]
\tag{7}
$$

**证明** 从反方向推导

$$
\begin{aligned}
& \mathbb{E}_{\pi_{\theta'}(\tau)}\left[ \sum^\infty_{t=0}\gamma^tA_{\pi_{\theta}}(S_{t},A_{t}) \right] \\
&= \mathbb{E}_{\pi_{\theta'}(\tau)}\left[ \sum^\infty_{t=0}\gamma^t\left(R(S_{t}, A_{t}) + \gamma V_{\pi_{\theta}}(S_{t+1}) - V_{\pi_{\theta}}(S_{t})\right) \right] \\
&= \mathbb{E}_{\pi_{\theta'}(\tau)}\left[ \sum^\infty_{t=0}\gamma^tR(S_{t}, A_{t}) + \sum^\infty_{t=1}\gamma^t V_{\pi_{\theta}}(S_{t}) - \sum^\infty_{t=0}\gamma^tV_{\pi_{\theta}}(S_{t}) \right] \\
&= \mathbb{E}_{\pi_{\theta'}(\tau)}\left[ \sum^\infty_{t=0}\gamma^tR(S_{t},A_{t}) - V_{\pi_{\theta}}(S_{0}) \right] \\
&= \mathbb{E}_{\pi_{\theta'}(\tau)}\left[ \sum^\infty_{t=0}\gamma^tR(S_{t},A_{t}) \right] - \mathbb{E}_{\pi_{\theta'}(\tau)}\left[ V_{\pi_{\theta}}(S_{0}) \right] \\
&= \mathbb{E}_{\pi_{\theta'}(\tau)}\left[ \sum^\infty_{t=0}\gamma^tR(S_{t},A_{t}) \right] - \mathbb{E}_{p(S_{0})}\left[ V_{\pi_{\theta}}(S_{0}) \right] \\
&= J(\theta') - J(\theta)
\end{aligned}
$$

第一步用到优势函数的定义和价值函数的性质。

$$
\begin{aligned}
A_{\pi_{\theta}}(S_{t},A_{t}) &= Q_{\pi_{\theta}}(S_{t},A_{t}) - V_{\pi_{\theta}}(S_{t})  \\
&= R(S_{t},A_{t})+\gamma\mathbb{E}_{P(S_{t+1}|S_{t},A_{t})}\left[ V_{\pi_\theta}(S_{t+1}) \right] - V_{\pi_{\theta}}(S_{t}) 
\end{aligned}
$$

**证毕** 。

接着可以推导出 $J(\theta')-J(\theta)$ 的以下关系。

$$
\begin{aligned}
& J(\theta') - J(\theta) \\
&= \mathbb{E}_{\pi_{\theta'}(\tau)}\left[ \sum^\infty_{t=0}\gamma^tA_{\pi_{\theta}}(S_{t},A_{t}) \right] \\
&= \sum^\infty_{t=0}\mathbb{E}_{\rho_{\theta'}(S_{t})}\left[ \mathbb{E}_{\pi_{\theta'}(A_{t}|S_{t})}\left[ \gamma^t A_{\pi_{\theta}} (S_{t},A_{t}) \right]  \right] \\
&= \sum^\infty_{t=0}\mathbb{E}_{\rho_{\theta'}(S_{t})}\left[ \mathbb{E}_{\pi_{\theta'}(A_{t}|S_{t})}\left[ \frac{\pi_{\theta'}(A_{t}|S_{t})}{\pi_{\theta}(A_{t}|S_{t})} \gamma^t A_{\pi_{\theta}} (S_{t},A_{t}) \right]  \right] \\
\end{aligned} \tag{8}
$$

最后一步用到重要性采样。重要性采样是通过一个概率分布的采样计算关于另一个概率分布的函数期望的方法。在这里，通过旧策略的采样计算关于新策略的（旧策略的）优势函数的期望。注意，内侧的期望是关于旧策略的分布的，外侧的期望是关于新策略的分布的。

另一方面，TRPO的目标函数是

$$
L(\theta';\theta) = \sum_{t=0}^\infty\mathbb{E}_{\rho_{\theta}(S_{t})}\left[ \mathbb{E}_{\pi_{\theta}(A_{t}|S_{t})} \left[ \frac{\pi_{\theta'}(A_{t}|S_{t})}{\pi_{\theta}(A_{t}|S_{t})}\gamma^tA_{\pi_{\theta}}(S_{t},A_{t}) \right]  \right]
\tag{9}
$$

内侧的期望是关于旧期望的分布的，外侧的期望也是关于旧策略的分布的。

比较式(8)中的 $J(\theta')-J(\theta)$ 和式(9)中的 $L(\theta';\theta)$ ，两者只有外侧的关于状态访问分布的期望不同。事实上，当新旧策略的策略函数 $\pi_{\theta'}(A_{t}|S_{t})$ 和 $\pi_{\theta}(A_{t}|S_{t})$ 接近时，新旧策略的状态访问分布 $\rho_{\theta'}(S_{t})$ 和 $\rho_{\theta}(S_{t})$ 也是接近的。因此，有以下关系成立：

$$
J(\theta') - J(\theta) \approx L(\theta';\theta)
$$

也就是说，在新旧策略接近的条件下最大化 $L(\theta';\theta)$ 等价于最大化 $J(\theta')-J(\theta)$ ，这就是TRPO的直观解释。

理论上，TRPO的单调性由以下定理简介保证。

**定理 1** 在约束条件下最大化目标函数 $L(\theta';\theta)$ （求解有约束的优化问题）。

$$
\begin{aligned}
\max_{\theta'}[L(\theta';\theta)]=\max_{\theta'}\left\{ \sum_{t=0}^\infty\mathbb{E}_{\rho_{\theta}(A_{t}|S_{t})}\left[ \mathbb{E}_{\pi_{\theta}(A_{t}|S_{t}) }\left[ \frac{\pi_{\theta'}(A_{t}|S_{t})}{\pi_{\theta}(A_{t}|S_{t})}\gamma^tA_{\pi_{\theta}}(S_{t},A_{t}) \right]  \right]  \right\} \\
\max_{t}\text{KL}[\pi_{\theta}(A_{t}|S_{t})|\pi_{\theta'}(A_{t}|S_{t})]\leq \delta
\end{aligned}
 \tag{10} 
$$

可以保证使 $J(\theta')-J(\theta)$ 增大。因为在约束条件下，$L(\theta';\theta)$ 是 $J(\theta')-J(\theta)$ 的下界。注意，这里约束条件中取KL散度的最大，而不是期望(2)。

TRPO在迭代过程中使用新旧两个策略，解决随机梯度方法中策略改进幅度过大所导致的训练不稳定的问题。$J(\theta')-J(\theta)$ 中的期望是关于新旧两个策略的，很难对其进行优化。相比，TRPO的目标函数 $L(\theta';\theta)$ 中的期望都是关于旧策略的，可以自然地对其进行优化。

### 1.4 具体算法

利用以下引理导出TRPO学习的优化问题(2)的具体形式。

**引理 2** 两个概率分布 $p_{\theta}(x)$ 和 $p_{\theta'}(x)$ 的KL散度可以近似表示为

$$
\text{KL}[p_{\theta}(x)|p_{\theta'}(x)] \approx \frac{1}{2} (\theta'-\theta)^TF(\theta)(\theta'-\theta) \tag{11}
$$

其中，$F(\theta)$ 是费舍尔信息矩阵（Fisher information matrix）。

$$
F(\theta) = -\mathbb{E}_{p_{\theta}(x)}[\nabla^2_{\theta}\log p_{\theta}(x)]
$$

**证明** KL散度关于 $\theta'$ 在 $\theta$ 的二阶泰勒展开是

$$
\begin{aligned}
\text{KL}[p_{\theta}(x)|p_{\theta'}(x)] &\approx \text{KL}[p_{\theta}(x)|p_{\theta}(x)] \\ &+(\theta'-\theta)^T\nabla_{\theta'}\text{KL}[p_{\theta}(x)|p_{\theta'}(x)]|_{\theta'=\theta} \\ &+(\theta'-\theta)^T\nabla_{\theta'}^2\text{KL}[p_{\theta}(x)|p_{\theta'}(x)]|_{\theta'=\theta}(\theta'-\theta)
\end{aligned}
$$

第一项是同一分布之间的KL散度，其值为 $0$ 。KL散度的一阶导数是

$$
\begin{aligned}
\nabla_{\theta'}\text{KL}[p_{\theta}(x)|p_{\theta'}(x)] &= \nabla_{\theta'}\mathbb{E}_{p_{\theta}(x)}[\log p_{\theta}(x)] - \nabla_{\theta'}\mathbb{E}_{p_{\theta}(x)}[\log p_{\theta'}(x)] \\
&= -\mathbb{E}_{p_{\theta}(x)}[\nabla_{\theta'}\log p_{\theta'}(x)] \\
&= -\int p_{\theta}(x)\nabla_{\theta'}\log p_{\theta'}(x)dx
\end{aligned}
$$

一阶导数在 $\theta'=\theta$ 取值为 $0$ 。

$$
\begin{aligned}
\nabla_{\theta'}\text{KL}[p_{\theta}(x)|p_{\theta'}(x)]|_{\theta'=\theta} &= -\int p_{\theta}(x) \nabla_{\theta}\log p_{\theta}(x)dx \\
&= -\int \nabla _{\theta}p_{\theta}(x)dx \\
&= -\nabla_{\theta}\int p_{\theta}(x)dx \\
&= -\nabla _{\theta}1=0
\end{aligned}
$$

KL散度的二阶导数是

$$
\nabla^2_{\theta'}\text{KL}[p_{\theta}(x)|p_{\theta'}(x)] = -\int p_{\theta}(x)\nabla^2_{\theta'}\log p(\theta')(x)dx
$$

二阶导数在 $\theta'=\theta$ 取值为费舍尔信息矩阵：

$$
\begin{aligned}
\nabla^2_{\theta'}\text{KL}[p_{\theta}(x)|p_{\theta'}(x)]|_{\theta'=\theta} &= -\int p_{\theta}(x)\nabla^2_{\theta}\log p_{\theta}(x)dx \\
&= F(\theta)
\end{aligned}
$$

以上的结果带入泰勒展开式得到近似公式。

**证毕** 。

TRPO的学习优化问题变成

$$
\begin{aligned}
\max_{\theta'}\mathbb{E}_{\rho_{\theta}(s)}\left[ \mathbb{E}_{\pi_{\theta}(a|s)} \left[ \frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}A_{\pi_{\theta}}(s,a) \right]  \right] \\
\mathbb{E}_{\rho_{\theta}(s)}\left[ \frac{1}{2} (\theta'-\theta)^TF(\theta)(\theta'-\theta) \right]\leq\delta \\
F(\theta) = -\mathbb{E}_{\pi_{\theta}(a|s)}\left[ \nabla^2_{\theta}\log \pi_{\theta}(a|s) \right] 
\end{aligned}
\tag{12}
$$

目标函数的梯度函数是

$$
\mathbb{E}_{\rho_{\theta}(s)}\left[ \mathbb{E}_{\pi_{\theta}(a|s)} \left[ \frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}\nabla_{\theta'}\log \pi_{\theta'}(a|s)A_{\pi_{\theta}}(s,a) \right]  \right] \tag{13} 
$$

这是一个二阶的优化问题，因为优化使用目标函数的一阶导数(梯度)，以及约束条件的二阶导数(黑塞矩阵)。

自然梯度下降（natural gradient descent）可以用于这个优化问题的求解。自然梯度下降是一种考虑参数空间几何结构的优化方法，通过利用费舍尔信息矩阵的逆矩阵调整梯度方向，以提高优化的效率和收敛性。TRPO进一步使用共轭梯度法（conjugate gradient）提升计算效率。这里不作介绍。

TRPO比起传统的策略梯度算法有多个优点。首先，策略更新的每一步都能保证策略性能的提升。其次，由于策略更新是限制在信任区间之内的，学习的优化过程有很高的稳定性。还有，由于训练稳定，随机梯度上升可以对数据进行多次遍历，有更高的样本使用效率。

## 2 PPO算法

PPO算法是TRPO算法的改进。PPO比TRPO更容易实现、有更高的计算效率，且经验上性能同等有效。本节概述PPO算法的基本想法，介绍PPO-Clip算法的具体实现。

### 2.1 算法概述

PPO也是策略梯度，特别是演员-评论员算法的改进，也属于在策略学习算法。TRPO算法使用二阶优化算法，实现复杂而且计算效率不高。PPO算法改为用一阶优化算法进行优化，以解决计算效率问题。具体地，PPO将原始的有约束最优化问题转换为无约束的最优化问题，用代理目标函数（surrogate objective function）代替目标函数，并且都通过随机梯度上升（一阶算法）进行优化。

PPO有两个变种，PPO-Penalty和PPO-Clip。前者在代理目标函数中使用惩罚项，后者在代理目标函数中使用截断函数，以近似表示优化约束条件。PPO-Penalty的优化问题是

