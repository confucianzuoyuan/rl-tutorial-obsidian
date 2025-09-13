
## 动机

重要性采样在抽样推理和强化学习（RL）中起着关键作用。在强化学习中，重要性采样利用先前从旧策略 $\pi'$ 中收集的样本来估计策略 $\pi$ 的价值函数。简单来说，计算采取某个行动的总奖励非常消耗算力。但是，如果新行动与旧行动相对接近，重要性采样允许我们基于旧策略的计算结果来计算新策略的奖励。

具体来说，使用强化学习中的蒙特卡洛方法，每当我们更新策略 $θ$ 时，我们都需要收集一个全新的轨迹来计算预期奖励。

![[重要性采样-1.excalidraw|1000]]

一条轨迹可能包含数百步，单次更新效率极低。使用重要性采样，我们只需重复使用旧样本即可重新计算总奖励。然而，当当前策略与旧策略偏差过大时，准确率就会下降。因此，我们需要定期重新同步两个策略。

在其他情况下，我们使用重要性采样来重写策略梯度方程，并使用它们来创建新的解决方案。

![[重要性采样-2.excalidraw|1000]]

例如，我们可以重新写一下我们的优化目标，但要限制策略的改变幅度不要太大。我们形式化了一个置信域的概念，我们相信该域内的近似值对于新策略仍然足够准确。

![[重要性采样-3.excalidraw|1000]]

通过避免采取过于激进的举措，我们可以更好地确保不会做出破坏训练进度的糟糕改变。随着我们不断改进策略，最终找到最优解。

## 什么是重要性采样？

重要性采样是一种估计 $f(x)$ 期望值的技术，其中 $x$ 服从数据分布 $p$ 。然而，我们不是从 $p$ 中采样，而是从 $q$ 中采样来计算结果。

$$
\mathbb{E}_p[f(x)] = \mathbb{E}_q\left(\frac{f(\mathbf{X})p(\mathbf{X})}{q(\mathbf{X})}\right)
$$

```ad-note
title: 证明

$$
\begin{aligned}
\mathbb{E}_p[f(\mathbf{X})] &= \int_\mathcal{X}f(x)p(x)dx \\
&= \int_\mathcal{X}f(x)p(x)\frac{q(x)}{q(x)}dx \\
&= \int_\mathcal{X}f(x)\frac{p(x)}{q(x)}q(x)dx \\
&= \mathbb{E}_q\left({f(\mathbf{X})\frac{p(\mathbf{X})}{q(\mathbf{X})}}\right)
\end{aligned}
$$
```

即我们使用抽样分布 $q$ 来估计 $p$ 的期望：

$$
\mathbb{E}_p[f(x)] \approx \frac{1}{n}\sum_{i=1}^n\mathbb{E}_q\left(\frac{f(\mathbf{X}_i)p(\mathbf{X}_i)}{q(\mathbf{X}_i)}\right),\quad\mathbf{X}_i\sim q.
$$

为了实现这一点，如果 $p(X_i)$ 不为 0 ， $q(X_i)$ 也不能为 0 。我们用一个例子来说明一下。

让我们首先计算 $\mathbb{E}_p[f(x)]$ 和 $\mathbb{E}_q[f(x)]$ 。它们应该是不同的。在下面的例子中，我们用分布 $p$ 计算 $\mathbb{E}_q[f(x)]$ 。

首先

$$
f(1) = 2,f(2) = 3, \text{以及}f(3)=4, \text{其它情况都为}0.
$$

假设概率分布 $p$ 和 $q$ 为

$$
\begin{aligned}
& p(x=1) = \frac{1}{3}, && p(x=2)=\frac{1}{3}, & p(x=3)=\frac{1}{3} \\
& q(x=1) = 0, && q(x=2)=\frac{1}{3}, & q(x=3)=\frac{2}{3} \\
\end{aligned}
$$

分别计算期望

$$
\mathbb{E}_p[f(x)] = 2\times\frac{1}{3}+3\times\frac{1}{3}+4\times\frac{1}{3}=3
$$

以及

$$
\mathbb{E}_q[f(x)] = 3\times\frac{1}{3}+4\times\frac{2}{3}=\frac{11}{3}
$$

现在，我们用分布 $p$ 重新加权的期望值。

$$
\begin{aligned}
\mathbb{E}_q[f(x)] &= \mathbb{E}_p\left[{f(x)\frac{q(x)}{p(x)}}\right] \\
&= \frac{1}{3}\times f(x=1)\times\frac{q(x=1)}{p(x=1)}+\frac{1}{3}\times f(x=2)\times\frac{q(x=2)}{p(x=2)} + \frac{1}{3}\times f(x=3)\times\frac{q(x=3)}{p(x=3)} \\
&= \frac{11}{3}
\end{aligned}
$$

现在，我们展示了如何使用不同的采样分布来计算期望。在强化学习中，我们会重用旧策略的采样结果来优化当前策略。

在抽样推断中， $p(x)$ 已经建模，计算每个 $x$ 的 $p(x)$ 并不困难。然而，存在一种误解，认为只要我们知道 $p$ 的方程，就知道如何轻松地创建代表分布 $p$ 的样本。这仅适用于众所周知的分布。一般来说，情况并非如此。因此，对于抽样推断，我们使用更易处理的分布 $q$ 来生成样本。

