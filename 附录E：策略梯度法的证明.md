
## 1. 策略梯度法的推导

当 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left\lbrack{G(\tau)}\right\rbrack$ 时，其梯度如下面的式子所示。

$$
\nabla_\theta{J_\theta}=\mathbb{E}_{\tau\sim\pi_\theta}\left\lbrack{\sum_{t=0}^TG(\tau)\nabla_\theta\log\pi_\theta(A_t|S_t)}\right\rbrack
$$

下面对上面的式子进行证明。

$$
\begin{split}
\nabla_\theta{J(\theta)} &= \nabla_\theta\mathbb{E}_{\tau\sim\pi_\theta}\left\lbrack{G(\tau)}\right\rbrack \\
&= \nabla_\theta\sum_\tau{\Pr(\tau|\theta)G(\tau)} \quad\text{（展开期望值）} \\
&= \sum_\tau\nabla_\theta(\Pr(\tau|\theta)G(\tau)) \quad\text{（将}\pi_\theta\text{移动到}\sum\text{中）} \\
&= \sum_\tau\left\lbrace{G(\tau)\nabla_\theta\Pr(\tau|\theta)+\Pr(\tau|\theta)\nabla_\theta{G(\tau)}}\right\rbrace \quad\text{（积的微分）} \\
&= \sum_\tau{G(\tau)\nabla_\theta\Pr(\tau|\theta)} \quad\text{（}\nabla_\theta{G(\tau)}\text{永远为0）} \\
&= \sum_\tau G(\tau)\Pr(\tau|\theta)\frac{\nabla_\theta\Pr(\tau|\theta)}{\Pr(\tau|\theta)} \quad\text{（乘以}{\frac{\Pr(\tau|\theta)}{\Pr(\tau|\theta)}}\text{）} \\
&= \sum_\tau G(\tau)\Pr(\tau|\theta)\nabla_\theta\log\Pr(\tau|\theta) \quad\text{（}\log\text{梯度的技巧）} \\
&= \mathbb{E}_{\tau\sim\pi_\theta}\left\lbrack{G(\tau)\nabla_\theta\log\Pr(\tau|\theta)}\right\rbrack
\end{split}\tag{1}
$$

这里对 “log梯度的技巧” 进行说明。这个技巧利用了以下等式。

$$
\nabla_\theta\log\Pr(\tau|\theta)=\frac{\nabla_\theta\Pr(\tau|\theta)}{\Pr(\tau|\theta)}
$$

> [!NOTE]
> $$\log(f(x))=\frac{f'(x)}{f(x)}$$

根据上面的式子，我们就知道

$$
\nabla_\theta\Pr(\tau|\theta)=\Pr(\tau|\theta)\nabla_\theta\log\Pr(\tau|\theta)
$$

这就是著名的 **log梯度的技巧** 。是机器学习领域常用的数学式的变形形式。

接下来，我们将利用以下等式进一步展开$(1)$。

$$
\begin{split}
\Pr(\tau|\theta) &= p(S_0)\pi_\theta(A_0|S_0)p(S_1|S_0,A_0)\cdots\pi_\theta(A_T|S_T)p(S_{T+1}|S_T,A_T) \\
&= p(S_0)\prod_{t=0}^T\pi_\theta(A_t|S_t)p(S_{t+1}|S_t,A_t)
\end{split}
$$

这里，$p(S_0)$ 表示初始状态 $S_0$ 的概率。上面的式子表明，得到轨迹 $\tau$ 的概率可以用初始状态的概率、策略以及下一个状态的迁移概率的乘积来表示。另外，我们可以用下面的式子来表示 $\log\Pr(\tau|\theta)$ 。

$$
\log\Pr(\tau|\theta)=\log{p(S_0)} + \sum_{t=0}^T\log{p(S_{t+1}|S_t,A_t)} + \sum_{t=0}^T\log\pi_\theta(A_t|S_t)
$$

由于 $\log xy = \log x + \log y$ ，所以可以像上面的式子那样表示为和的形式。基于上面的式子，可以将 $\nabla_\theta{\log\Pr(\tau|\theta)}$ 展开为如下形式。

$$
\begin{split}
\nabla_\theta\log\Pr(\tau|\theta) &= \nabla_\theta\left\lbrace\log{p(S_0)} + \sum_{t=0}^T\log{p(S_{t+1}|S_t,A_t)} + \sum_{t=0}^T\log\pi_\theta(A_t|S_t)\right\rbrace \\
&= \nabla_\theta\sum_{t=0}^T\log\pi_\theta(A_t|S_t)
\end{split}
$$

$\nabla_\theta$ 是对 $\theta$ 的梯度。与 $\theta$ 无关的元素的梯度 $\nabla_\theta\log p(S_0)$ 和 $\nabla_\theta\sum_{t=0}^T\log{p(S_{t+1}|S_t,A_t)}$ 为 0 。因此，从上面的式子可以得到下列式子。

$$
\begin{split}
\nabla_\theta{J(\theta)}&=\mathbb{E}_{\tau\sim\pi_\theta}\left\lbrack{G(\tau)\nabla_\theta\log\Pr(\tau|\theta)}\right\rbrack \\
&= \mathbb{E}_{\tau\sim\pi_\theta}\left\lbrack{\sum_{t=0}^TG(\tau)\nabla_\theta\log\pi_\theta(A_t|S_t)}\right\rbrack
\end{split}
$$

这样我们就完成了 $\nabla_\theta{J(\theta)}$ 的推导。

## 2. 基线的推导

$$
\begin{split}
\nabla_\theta J(\theta) &= \mathbb{E}_{\tau\sim\pi_\theta}\left\lbrack{\sum_{t=0}^TG_t\nabla_\theta\log\pi_\theta(A_t|S_t)}\right\rbrack \\
&= \mathbb{E}_{\tau\sim\pi_\theta}\left\lbrack{\sum_{t=0}^T(G_t-b(S_t))\nabla_\theta\log\pi_\theta(A_t|S_t)}\right\rbrack \\
\end{split}
$$

如上面的式子所示，我们可以使用 $G_t-b(S_t)$ 代替 $G_t$ 。$b(S_t)$ 是 **任何函数** ，我们称之为“基线”。下面进行上式的推导。

首先，证明以下式子成立。

$$
\mathbb{E}_{x\sim P_\theta}\left\lbrack{\nabla_\theta\log P_\theta(x)}\right\rbrack = 0 \tag{1}
$$

这里假设随机变量 $x$ 是基于概率分布 $P_\theta(x)$ 生成的。$P_\theta(x)$ 会根据参数 $\theta$ 改变概率分布的形状。此时有以下式子成立。

$$
\sum_xP_\theta(x)=1
$$

由于 $P_\theta(x)$ 是概率分布，因此所有 $x$ 的值的和为 1 。然后，求这个式子的梯度。

$$
\nabla_\theta\sum_xP_\theta(x)=\nabla_\theta 1 = 0
$$

接下来，使用 $\log$ 梯度的技巧将式子展开，过程如下所示。

$$
\begin{split}
0 &= \nabla_\theta\sum_xP_\theta(x) \\
&= \sum_x\nabla_\theta P_\theta(x) \\
&= \sum_xP_\theta(x)\nabla_\theta\log P_\theta(x) \\
&= \mathbb{E}_{x\sim P_\theta}\left\lbrack{\nabla_\theta\log P_\theta(x)}\right\rbrack
\end{split}
$$

证明(1)完毕。接下来将证明的式子用于我们的问题。具体来说，用 $A_t$ 代替(1)中的 $x$ ，然后使用 $\pi_\theta(\cdot|S_t)$ 代替 $P_\theta(\cdot)$ 。这样就可以得到以下式子。

$$
\mathbb{E}_{A_t\sim\pi_\theta}\left\lbrack{\nabla_\theta\log\pi_\theta(A_t|S_t)}\right\rbrack = 0
$$

上面的式子是对 $A_t$ 的期望值。因此，我们可以像下面的式子那样，将任何函数 $b(S_t)$ 放入期望值中。

$$
\mathbb{E}_{A_t\sim\pi_\theta}\left\lbrack{b(S_t)\nabla_\theta\log\pi_\theta(A_t|S_t)}\right\rbrack = 0 \tag{2}
$$

$b(S_t)$ 是以 $S_t$ 为参数的函数，即使 $A_t$ 发生变化，它的值也不会改变。由于式子(2)是对 $A_t$ 的期望值，因此即使在期望值中加入函数 $b(S_t)$ ，等式也成立。

> [!NOTE]
> 动作 $A_t$ 的变化会导致收益 $G_t$ 的变化，因此以下式子不成立。
> $$\mathbb{E}_{A_t\sim\pi_\theta}\left\lbrack{G_t\nabla_\theta\log\pi_\theta(A_t|S_t)}\right\rbrack = 0$$

式子(2)在整个 $t=0\sim T$ 的范围都成立，所以可以得到以下式子。

$$
\mathbb{E}_{A_t\sim\pi_\theta}\left\lbrack{\sum_{t=0}^Tb(S_t)\nabla_\theta\log\pi_\theta(A_t|S_t)}\right\rbrack = 0
$$

所以基线证明完毕。