
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

$$
\begin{aligned}
\max_{\theta'}[L(\theta';\theta)]=\max_{\theta'}\mathbb{E}_{\rho_{\theta}(s)}[\mathbb{E}_{\pi_{\theta}(a|s)}[L_{p}(\theta';\theta,s,a)]] \\
L_{p}(\theta';\theta,s,a)=\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}A_{\pi_{\theta}}(s,a)-\beta \text{KL}(\pi_{\theta'}(a|s)|\pi_{\theta}(a|s))
\end{aligned}\tag{14}
$$

PPO-Clip的优化问题是

$$
\begin{aligned}
\max_{\theta'}[L(\theta';\theta)]=\max_{\theta'}\mathbb{E}_{\rho_{\theta}(s)}[\mathbb{E}_{\pi_{\theta}(a|s)}[L_{c}(\theta';\theta,s,a)]] \\
L_{c}(\theta';\theta,s,a)=\min\left\{ \frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}A_{\pi_{\theta}}(s,a),\text{clip}\left( \frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)},1-\epsilon,1+\epsilon\right)A_{\pi_{\theta}}(s,a)  \right\} 
\end{aligned}\tag{15}
$$

其中，clip是截断函数。

PPO-Clip实现更简单，经验上与PPO-Penalty性能同等有效，在现实中更常用。当提及PPO时一般指PPO-Clip。下面只对PPO-Clip做讲解。

### 2.2 PPO-Clip

PPO-Clip也是迭代算法，由数据采样、策略评估、策略改进三步组成。在迭代过程中，使用旧策略 $\pi_{\theta}(a|s)$ 和新策略 $\pi_{\theta'}(a|s)$ 。用旧策略进行数据采样，计算旧策略的优势函数，对新策略通过随机梯度上升进行改进。形式化为解有约束的优化问题，其中的目标函数是截断的目标函数。

截断的目标函数形式复杂，但概念并不复杂。其想法是对新策略的更新幅度进行限制，使得新旧策略的概率比值 $r_{\theta'}(s,a)$ 不会过大或过小，对应着TRPO的信任区间。

PPO-Clip算法在每一步使用随机梯度上升对目标函数(15)进行优化。对于截断函数clip，当输入 $r$ 在区间 $[1-\epsilon,1+\epsilon]$ 内时，将其直接输出；当输入 $r$ 在区间外时，将其截断为 $1+\epsilon$ 或 $1-\epsilon$ 并输出。

$$
\text{clip}(r,1-\epsilon,1+\epsilon)=\begin{cases}
r, & 1-\epsilon\leq r\leq_{1}+\epsilon \\
1-\epsilon, & r<1-\epsilon \\
1+\epsilon, & r>1+\epsilon
\end{cases}
$$

其中，$\epsilon$ 是超参数，经常取 $0.2$ 。

表(1)给出截断的目标函数的函数表。图(1)给出截断的目标函数的函数图。

```ad-note
title: 表 1 截断的目标函数


|比值 $r$ | 优势 $A$ | 目标 $L_c$ | 是否截断 | 梯度 $\nabla L_c$ 是否为 $0$ |
| --- | --- | --- | --- | --- |
| $1-ϵ\leq r\leq 1+ϵ$ |  $A\geq 0$ | $rA$ | 否 | 否 |
| $1-ϵ\leq r\leq 1+ϵ$ |  $A<0$ | $rA$ | 否 | 否 |
| $r<1-ϵ$ |  $A\geq 0$ | $rA$ | 否 | 否 |
| $r<1-ϵ$ |  $A<0$ | $(1-ϵ)A$ | 是 | 是 |
| $r>1+ϵ$ |  $A\geq 0$ | $(1+ϵ)A$ | 是 | 是 |
| $r>1+ϵ$ |  $A<0$ | $rA$ | 否 | 否 |
```

```ad-tip
title: 图 1 截断的目标函数

![[PPO-Ratio.excalidraw|1000]]
```

当优势函数大于等于 $0$ 时，让新策略 $\pi_{\theta'}(a|s)$ 对旧策略 $\pi_{\theta}(a|s)$ 的改进，其比值不超过上界 $1+\epsilon$ ，新策略的策略函数不会变得过大；当优势函数小于零时，让新策略 $\pi_{\theta'}(a|s)$ 对旧策略 $\pi_{\theta}(a|s)$ 的改进，其比值不超过下界 $1-\epsilon$ ，新策略的策略函数不会变得过小。当目标函数被截断时，对应的梯度函数为 $0$ ，对策略函数不做更新。

也就是说，PPO-Clip实际只有两种情况。目标函数的计算只需要考虑这两种情况。当 $A\geq 0$ 时，

$$
L_{c}(\theta';\theta,s,a)=\min\left\{ \frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)} ,1+\epsilon \right\}A_{\pi_{\theta}}(s,a) \tag{16}
$$

当 $A<0$ 时，

$$
L_{c}(\theta';\theta,s,a)=\min\left\{ \frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)} ,1-\epsilon \right\}A_{\pi_{\theta}}(s,a) \tag{17}
$$

### 2.3 具体算法

下面给出PPO-Clip的具体算法。假设模型是有限期MDP。

```ad-danger
title: 算法 1 （PPO-Clip）

输入：模型未知的有限期MDP。
输出：估计的最优策略 $\pi_{\hat{\theta'}}(a|s)$ 。
超参数：迭代次数 $K$ ，折扣因子 $γ$ 。
{
$\quad$ 初始化策略函数 $π_{θ'}(a|s)$ 的参数 $θ^{(0)}$
$\quad$ 初始化价值函数 $V_𝜙(s)$ 的参数 $𝜙^{(0)}$
$\quad$ $\text{for}\quad(k=1,2,\cdots,K)\quad\text{do}\quad \{$
$\quad\quad$ 根据当前策略 $π_{θ^{(k)}}(a|s)$ 与环境交互得到轨迹集合 $\mathcal{T}_k=\{\tau_1,\tau_2,\cdots,\tau_N\}$
$\quad\quad$ 计算轨迹 $\tau_n$ 的第 $t$ 步的剩余回报 $\hat{G}_{n,t}$
$\quad\quad$ 计算轨迹 $\tau_n$ 的第 $t$ 步的优势 $\hat{A}_{\pi_\theta}(S_{n,t},A_{n,t})$

$$
\hat{A}_{\pi_\theta}(S_{n,t},A_{n,t}) = R(S_{n,t},A_{n,t})+γV_𝜙^{(k)}(S_{n,t+1})-V_𝜙^{(k)}(S_{n,t})
$$

$\quad\quad$ 使用随机梯度上升法，更新策略函数的参数

$$
θ^{(k+1)} = \arg\max_{θ'}\frac{1}{NT}\sum_{n=1}^N\sum_{t=0}^{T-1}L_c(θ';θ^{(k)},S_{n,t},A_{n,t})
$$

$\quad\quad$ 使用随机梯度下降法，更新价值函数的参数

$$
𝜙^{(k+1)}
$$

$\quad$ }
}
```

PPO算法兼具TRPO算法的有点，同时比TRPO算法更简单，实现更容易，训练效率更高。

## 3. 大语言模型的应用

### 3.1 LLM概述

2022年11月OpenAI公布的ChatGPT，以及后续的GPT4，还有其他公司和组织开发的各种大语言模型，在自然语言理解和自然语言生成等任务上展现出了接近或部分超过人类的能力，标志着人工智能进入了新的时代。

大语言模型（large language model，LLM）是概率生成模型，生成单词或词元（token）的序列。通常采用GPT模型架构，也就是Transformer的解码器；学习和预测都是自回归过程。将所有的自然语言处理任务都转换为词元序列生成问题，包括生成、问答、改写、摘要、翻译、分类、对话。所有这些任务的形式都是对给定的输入产生对应的输出，而输入和输出都由词元的序列表示。输入包括上下文（context）和指令（prompt），输出是反应（response）。上下文可以是系统指令、模型与用户多次交互的历史记录。为了简洁起见，以下介绍有时会省略对上下文的提及，仅提及指令本身。

LLM的训练一般分三个步骤：预训练（pre-training）、监督微调（supervised fine-tuning，SFT）、人类反馈强化学习（reinforcement learning from human feedback，RLHF）。有时在预训练后面加入持续学习（continual training）。

1. 预训练：用大规模无标注文本数据进行语言建模，主要学习语言的统计规律，语法、语义和语用知识。
2. 持续学习（可选）：在预训练的基础上，用特定任务或特定领域数据进一步训练模型，主要为了提高执行任务的能力和扩充领域的知识。
3. SFT：用下游任务的高质量标注数据微调模型，使其适应下游任务；或者用高质量标注数据优化模型，保证行为的安全和与人类偏好的对齐。
4. RLHF：在SFT的基础上，通过人的反馈（如排序、打分）训练一个奖励模型，然后用这个奖励模型指导强化学习，例如PPO算法，进一步提升模型完成各种下游任务的能力，提高安全性和对齐能力。

### 3.2 预训练

LLM的预训练就是语言建模（自回归过程）的学习，与GPT的预训练方法完全相同。学习是无监督学习，训练数据中的每一个词元序列对应一个句子或者一段文本，预测下一个词元的生成概率，计算整个词元序列的概率。通过最小化训练数据中所有词元序列的概率，优化模型的参数。也就是进行极大似然估计，等价于对词元序列进行数据压缩。

模型表示的词元序列的概率是

$$
P_{\theta}(w)=\prod_{t}P_{\theta}(w_{t}|w_{<t})\tag{18}
$$

其中，$w$ 表示一个词元序列，$w_{t}$ 是第 $t$ 个位置的词元，$w_{<t}$ 是第 $t$ 个位置之前的词元序列，$0$ 是Transformer的参数。目标是最小化负对数似然函数或交叉熵

$$
L(\theta)=-\mathbb{E}_{w\sim P_{\text{data}}(w)}\sum_{t}\log P_{\theta}(w_{t}|w_{<t})\tag{19}
$$

其中，$P_{\text{data}}(w)$ 是词元序列数据的分布。

LLM的预训练使用大量的语料。比如，GPT-3使用了包含4950亿词元的语料，有网页、书籍、百科等。模型的参数是1750亿。

### 3.3 SFT

SFT是指在预训练的基础上的模型的监督学习。这里把LLM看作强化学习的智能体。智能体与环境交互的过程就是词元序列生成的过程。词元序列生成对给定上下文和用户指令，产生系统的回复。自然语言处理的各种任务包括理解和生成都转化为词元序列生成。模型写作

$$
\pi_{\theta}(y|x) = \prod_{t}\pi_{\theta}(y_{t}|x,y_{<t})\tag{20}
$$

其中，$x$ 表示指令，$y$ 表示回复，$y_{t}$ 是第 $t$ 个位置的词元，$y_{<t}$ 是第 $t$ 个位置之前的词元序列，$0$ 是Transformer的参数。

强化学习框架下，上下文和用户指令、目前为止已生成的词元序列表示状态，生成的下一个词元表示动作。生成词元时得到一个奖励，一般在生成最后一个词元也就是完成回复时得到一个实数值的奖励，而在其他生成时奖励为 $0$ ，奖励值的大小表示回复的合理性。奖励函数的设计是应用时需要考虑的重要问题。策略是状态到动作的条件概率分布

$$
\pi:(x,y_{<t})\to y_{t}
$$

奖励是状态到动作的函数

$$
R:(x,y_{<t})\to r_{t}
$$

状态转移是确定性的

$$
P:(x,y_{<t}),y_{t}\to(x,y_{t})
$$

这里 $x,y_{<t}$ 是指令和第 $t$ 个位置之前的词元序列，$y_{t}$ 是第 $t$ 个位置生成的词元，$x,y_{t}$ 是指令和第 $t$ 个位置为止的词元序列，$r_{t}$ 是在第 $t$ 个位置得到的奖励。

SFT相当于强化学习中的模仿学习或者行为克隆。学习通过最小化负对数似然函数进行：

$$
L(\theta)=-\mathbb{E}_{(x,y^*)\sim P_{\text{data}}(x,y)}\sum_{t}\log \pi_{\theta}(y_{t}^*|x,y^*_{<t}) \tag{21}
$$

其中，$y^*$ 是人标注的回复，$P_{\text{data}}(x,y)$ 是指令和回复数据的分布。通过基于人的标注数据的训练可以使模型与人类偏好对齐。

### 3.4 RLHF

RLHF主要分两部分，奖励模型的学习和策略模型的学习，也就是LLM的微调。

1. 奖励模型学习

学习奖励模型RM，以引导强化学习训练。RM的质量对RLHF的效果有很大影响。一个好的RM应该能够较好地拟合人类偏好，给出合理的奖励值。

首先，标注偏好数据。给定一个指令，让模型生成 $k$ 个回复，然后让人对这些回复进行排序或打分。每 $k$ 个回复两两构建成多个四元组，表示为 $(x,y_{1},y_{2},l)$ ，其中 $x$ 是指令，$y_{1}$ 和 $y_{2}$ 是两个回复，$l$ 是一个二元标签，如果 $y_{1}$ 比 $y_{2}$ 更合理，则 $l=1$ ，否则 $l=0$ 。

然后，使用训练数据学习一个二分类模型，作为RM。RM的输入是指令 $x$ 和回复 $y$ ，输出是标量 $r$ ，表示奖励。目标是最小化分类误差

$$
L(\phi) = -\frac{1}{\left( \begin{array}{c} k \\ 2 \end{array} \right)}
\mathbb{E}_{x\sim P_{\text{data}}(x),(y_{1},y_{2})\sim \pi_{\theta}(y|x)}\left[ \log \sigma(R_{\phi}(x,y_{1}) - R_{\phi}(x,y_{2})) \right]
\tag{22}
$$

其中，$\left( \begin{array}{c} k \\ 2 \end{array} \right)$ 表示所有可能的回复对的数量，$P_{data}(x)$ 是指令数据的分布，$\sigma$ 是 $S$ 型函数，$R_{\phi}(x,y_{1})$ 和 $R_{\phi}(x,y_{2})$ 是奖励函数，$\phi$ 是RM的参数。假设 $y_{1}$ 比 $y_{2}$ 更合理。

2. 策略模型学习

学习策略模型，也就是进行LLM微调，进一步提高LLM与人类偏好对齐的能力。一般使用PPO算法，也有DPO（direct preference optimization）等其他算法。

LLM作为智能体与环境交互，并在交互中得到反馈。在每一轮交互中，LLM基于给定的指令生成回复。这个过程通过多次调用LLM完成，每次生成一个词元，直到产生结束字符。在回复生成的每个位置，调用奖励模型评估并计算累计奖励。

学习的目标是最大化以下价值函数，以促使LLM生成合理的回复，同时学习的策略与基准策略不要偏离太远。

$$
J(\theta)=\mathbb{E}_{x\sim P_{\text{data}}(x),y\sim \pi_{\theta}(y|x)}\left[ R_{\phi}(x,y) - \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{sft}}(y|x)} \right]\tag{23} 
$$

其中，第1项是奖励函数，第2项是正则化项，表示当前策略和SFT模型的策略的KL散度，$\beta$是系数。

一般使用PPO算法来完成策略模型的学习。这里省去细节的介绍。

### 3.5	LLM的特点

LLM的强大能力可以从三个方面理解：规模定律、语言建模技巧、强化学习框架。

规模定律（scaling law）是指随着模型规模的不断扩大，语言模型在预测下一个词元（生成下一个词元）的准确率也会不断提升的现象。LLM能够看似理解语言，其中一个非常关键的原因就是规模效应。它的能力是建立在海量数据、大规模模型和强大算力的基础之上的。语言理解能力的提升在很大程度上来自于规模效应。

语言模型的基本机制是给定上文产生最有可能的下文。事实上，对于特定的上下文和指令，可能的合理回复会局限在一定的范围之内，甚至是很小的范围内。上下文越长这个趋势就越明显。LLM在训练和使用过程中最大限度地利用了这一特点。LLM有效地使用Transformer架构、自回归生成机制、大规模模型、海量数据训练，当有充分的上下文时，一般能够生成合理的回答。因此，构建充分且有效的上下文成为训练和使用LLM时最关键的技巧。在训练过程中，需要做好数据工程，准备好高质量数据进行预训练、SFT和RLHF。在使用过程中，也需要提供有效的指令，进行细致的指令工程。

LLM也是强化学习智能体，具体的是智能体的策略模型。其中，强化学习的动作就是LLM生成的回复，状态是迄今为止LLM与用户的对话内容，而奖励则表示对话的合理性。LLM学习的目标是在与用户的对话中学习到最优策略，从而在对话中获得最大的期望累积奖励，也就是进行合理的对话。实际将对话数据看作强化学习的数据，将SFT视为行为克隆，将RLHF视为奖励模型和策略模型学习。在强化学习过程中，Transformer的参数得到进一步调整，以产生从强化学习角度更优的模型。这时，语言建模和强化学习被统一在一起。

