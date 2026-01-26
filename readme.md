## 环境

``` bash
python=3.8.2
numpy==1.22.0
matplotlib==3.5.0
gym[classic_control]==0.23.0
Scipy==1.8.0
Pygame==2.1.2
Torch==1.8.0
```

## 非稳态 bandit 条件下，Qs 收敛证明

$$
\begin{aligned}
Q_{n} & =Q_{n-1}+\alpha_{n-1}\left[R_{n-1}-Q_{n-1}\right] \\
& =\left(1-\alpha_{n-1}\right) Q_{n-1}+\alpha_{n-1} R_{n-1} \\
& =\left(1-\alpha_{n-1}\right)\left[\left(1-\alpha_{n-2}\right) Q_{n-2}+\alpha_{n-2} R_{n-2}\right]+\alpha_{n-1} R_{n-1} \\
& =\left(1-\alpha_{n-1}\right)\left(1-\alpha_{n-2}\right) Q_{n-2}+\left(1-\alpha_{n-1}\right) \alpha_{n-2} R_{n-2}+\alpha_{n-1} R_{n-1} \\
& =\ldots \\
& =\prod_{i=1}^{n-1}\left(1-\alpha_{i}\right) Q_{1}+\sum_{i=1}^{n-1}\left[\alpha_{i} \prod_{j=i+1}^{n-1}\left(1-\alpha_{j}\right) R_{i}\right] \\
& =\sum_{i=1}^{n-1}\left[\alpha_{i} \prod_{j=i+1}^{n-1}\left(1-\alpha_{j}\right) R_{i}\right] \quad\left(\text { 设 } Q_{1}=0\right) \\
& =\sum_{i=1}^{n-1} w_{i} R_{i} \quad\left(\text { 其中 } w_{i}=\alpha_{i} \prod_{j=i+1}^{n-1}\left(1-\alpha_{j}\right)\right) \\
& =\sum_{i=1}^{n-1} w_{i}\left(Q^{*}_{i}+X_{i}\right) \quad\left(\text { 其中 } R_{i}=Q^{*}_{i}+X_{i}, Q^{*}_{i} \text{为当前环境下的期望奖励}, X_{i} \text{为零均值噪声}\right)
\end{aligned}
$$

**当 $\alpha_{n-1}=\frac{1}{n}$ 时，有：**
$$
\begin{aligned}
w_{i} & =\alpha_{i} \prod_{j=i+1}^{n-1}\left(1-\alpha_{j}\right) \\
& =\frac{1}{i+1} \prod_{j=i+1}^{n-1}\left(1-\frac{1}{j+1}\right) \\
& =\frac{1}{i+1} \prod_{j=i+1}^{n-1} \frac{j}{j+1} \\
& =\frac{1}{i+1} \cdot \frac{i+1}{n} \\
& =\frac{1}{n} \\
\\
\Rightarrow Q_{n} & =\sum_{i=1}^{n-1} \frac{1}{n}\left(Q^{*}_{i}+X_{i}\right)
\end{aligned}
$$

**当 $\alpha_{n-1}$ 为定值时，有：**
$$
\begin{aligned}
w_{i} & =\alpha_{i} \prod_{j=i+1}^{n-1}\left(1-\alpha_{j}\right) \\
& =\alpha \prod_{j=i+1}^{n-1}(1-\alpha) \\
& =\alpha(1-\alpha)^{n-1-i} \\
\\
\Rightarrow Q_{n} & =\sum_{i=1}^{n-1} \alpha(1-\alpha)^{n-1-i}\left(Q^{*}_{i}+X_{i}\right)
\end{aligned}
$$

由此可见，
- 当 $\alpha_{n-1}=\frac{1}{n}$ 时，Qs 对所有过去的奖励均等加权，适用于稳态环境下的 bandit 问题
- 当 $\alpha_{n-1}$ 为定值时，Qs 对近期奖励赋予更大权重，权重按照指数衰减，适用于非稳态环境下的 bandit 问题

## 贝尔曼方程

$$
\begin{aligned}
G_{t} & = R_{t} + \gamma R_{t+1} + \gamma^{2} R_{t+2} + \gamma^{3} R_{t+3} + \ldots \\
& = R_{t} + \gamma\left(R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + \ldots\right) \\
& = R_{t} + \gamma G_{t+1} \\
& \text{其中} \\
& R_{t} \text{为 t 时刻的奖励} \\
& \gamma \in[0,1] \text{为折扣因子} \\
& G_{t} \text{为 t 时刻以及之后的累计折扣奖励} \\
\end{aligned}
$$

**状态价值函数的贝尔曼方程**
$$
\begin{aligned}
v_{\pi}(s) & =\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right] \\
& =\mathbb{E}_{\pi}\left[R_{t}+\gamma G_{t+1} \mid S_{t}=s\right] \\
& =\mathbb{E}_{\pi}\left[R_{t} \mid S_{t}=s\right] + \gamma \mathbb{E}_{\pi}\left[G_{t+1} \mid S_{t}=s\right] \\
& \text{其中} \\
& v_{\pi}(s) \text{为策略} \pi \text{在状态 } s \text{ 下的状态价值函数} \\
\\
\mathbb{E}_{\pi}\left[R_{t} \mid S_{t}=s\right] & = \sum R(s, a, s') * P(a, s' | s, \pi) \\
& = \sum_{a} \pi(a | s) \sum_{s'} P(s' | s, a) R(s, a, s') \\
& = \sum_{a, s'} \pi(a | s) P(s' | s, a) R(s, a, s') \\
& 其中 \\
& R(s,a,s') 表示在状态 s 下采取动作 a，并转移到状态 s' 时所获得的奖励。\\
\\
\mathbb{E}_{\pi}\left[G_{t+1} \mid S_{t}=s\right] & = \sum P(a, s' | s, \pi) \mathbb{E}_{\pi}\left[G_{t+1} \mid S_{t+1}=s'\right] \\
& = \sum_{a} \pi(a | s) \sum_{s'} P(s' | s, a) v_{\pi}(s') \\
& = \sum_{a, s'} \pi(a | s) P(s' | s, a) v_{\pi}(s') \\
\\
\Rightarrow v_{\pi}(s) & = \sum_{a, s'} \pi(a | s) P(s' | s, a) \left[R(s, a, s') + \gamma v_{\pi}(s')\right]
\end{aligned}
$$

**动作价值函数的贝尔曼方程**
$$
\begin{aligned}
q_{\pi}(s, a) & =\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s, A_{t}=a\right] \\
& =\mathbb{E}_{\pi}\left[R_{t}+\gamma G_{t+1} \mid S_{t}=s, A_{t}=a\right] \\
& =\mathbb{E}_{\pi}\left[R_{t} \mid S_{t}=s, A_{t}=a\right] + \gamma \mathbb{E}_{\pi}\left[G_{t+1} \mid S_{t}=s, A_{t}=a\right] \\
& \text{其中} \\
& q_{\pi}(s, a) \text{为策略} \pi \text{在状态 } s \text{ 下采取动作 } a \text{ 的动作价值函数} \\
\\
\mathbb{E}_{\pi}\left[R_{t} \mid S_{t}=s, A_{t}=a\right] & = \sum_{s'} P(s' | s, a) R(s, a, s') \\
& 其中 \\
& R(s,a,s') 表示在状态 s 下采取动作 a，并转移到状态 s' 时所获得的奖励。\\
\\
\mathbb{E}_{\pi}\left[G_{t+1} \mid S_{t}=s, A_{t}=a\right] & = \sum P(s' | s, a, \pi) \mathbb{E}_{\pi}\left[G_{t+1} \mid S_{t+1}=s'\right] \\
& = \sum_{s'} P(s' | s, a) v_{\pi}(s') \\
& = \sum_{s'} P(s' | s, a) \sum_{a'} \pi(a' | s') q_{\pi}(s', a') \\
\Rightarrow q_{\pi}(s, a) & = \sum_{s'} P(s' | s, a) \left[R(s, a, s') + \gamma v_{\pi}(s')\right] \\
& = \sum_{s'} P(s' | s, a) \left[R(s, a, s') + \gamma \sum_{a'} \pi(a' | s') q_{\pi}(s', a')\right]
\end{aligned}
$$

**总结**
状态价值函数描述了在给定状态下，遵循特定策略所能获得的预期回报。而动作价值函数则在此基础上对当前动作进行了条件化，描述了在特定状态下采取特定动作后，遵循该策略所能获得的预期回报。
也就是说状态价值函数描述的是在给定状态下，全局动作域中所有动作的期望回报，而动作价值函数则具体到某一个动作的期望回报。
所以 **状态价值函数 = 动作价值函数 在 全局动作域 上的加权平均，权重即为策略在该状态下选择各动作的概率分布**。

## 动态规划法求解状态价值函数的贝尔曼方程收敛性和唯一性证明

**状态价值函数的贝尔曼方程**
$$
\begin{aligned}
G_{t} & = R_{t} + \gamma G_{t+1} \\
v_{\pi}(s) & =\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right] \\
& =\mathbb{E}_{\pi}\left[R_{t}+\gamma G_{t+1} \mid S_{t}=s\right] \\
& =\mathbb{E}_{\pi}\left[R_{t} \mid S_{t}=s\right] + \gamma \mathbb{E}_{\pi}\left[G_{t+1} \mid S_{t}=s\right] \\
& = \sum_{a, s'} \pi(a | s) P(s' | s, a) \left[R(s, a, s') + \gamma v_{\pi}(s')\right]
\end{aligned}
$$

**状态转移表达式**
$$
\begin{aligned}
v_{k+1}(s) & = \sum_{a, s'} \pi(a | s) P(s' | s, a) \left[R(s, a, s') + \gamma v_{k}(s')\right]
\end{aligned}
$$

**收敛性证明**
$$
\begin{aligned}
\|v_{k+1} - v_{\pi}\|_{\infty} & = \max_{s} |v_{k+1}(s) - v_{\pi}(s)| \\
& = \max_{s} \left| \sum_{a, s'} \pi(a | s) P(s' | s, a) \left[R(s, a, s') + \gamma v_{k}(s')\right] - \sum_{a, s'} \pi(a | s) P(s' | s, a) \left[R(s, a, s') + \gamma v_{\pi}(s')\right] \right| \\
& = \max_{s} \left| \sum_{a, s'} \pi(a | s) P(s' | s, a) \gamma \left[v_{k}(s') - v_{\pi}(s')\right] \right| \\
& \leq \max_{s} \sum_{a, s'} \pi(a | s) P(s' | s, a) \gamma \left|v_{k}(s') - v_{\pi}(s')\right| \\
& \leq \max_{s} \sum_{a, s'} \left[ \pi(a | s) P(s' | s, a) (\gamma \max_{s'} \left|v_{k}(s') - v_{\pi}(s')\right|) \right] \\
& = (\gamma \max_{s'} \left|v_{k}(s') - v_{\pi}(s')\right|) \max_{s} \sum_{a, s'} \pi(a | s) P(s' | s, a) \\
& = \gamma \max_{s'} \left|v_{k}(s') - v_{\pi}(s')\right| \\
& = \gamma \|v_{k} - v_{\pi}\|_{\infty} \\
\\
\Rightarrow \|v_{k+1} - v_{\pi}\|_{\infty} & \leq \gamma^{k+1} \|v_{0} - v_{\pi}\|_{\infty} \\
& \text{由于 } 0 \leq \gamma < 1, \text{ 当 } k \to \infty, \gamma^{k+1} \to 0 \\
& \Rightarrow \lim_{k \to \infty} \|v_{k+1} - v_{\pi}\|_{\infty} = 0 \\
& \Rightarrow \lim_{k \to \infty} v_{k}(s) = v_{\pi}(s)
\end{aligned}
$$

**唯一性证明**
$$
\begin{aligned}
& \text{假设存在两个不同的状态价值函数 } v_{\pi} \text{ 和 } u_{\pi}, \text{ 满足贝尔曼方程} \\
v_{\pi}(s) & = \sum_{a, s'} \pi(a | s) P(s' | s, a) \left[R(s, a, s') + \gamma v_{\pi}(s')\right] \\
u_{\pi}(s) & = \sum_{a, s'} \pi(a | s) P(s' | s, a) \left[R(s, a, s') + \gamma u_{\pi}(s')\right] \\
\\
\Rightarrow \|v_{\pi} - u_{\pi}\|_{\infty} & = \max_{s} |v_{\pi}(s) - u_{\pi}(s)| \\
& = \max_{s} \left| \sum_{a, s'} \pi(a | s) P(s' | s, a) \left[R(s, a, s') + \gamma v_{\pi}(s')\right] - \sum_{a, s'} \pi(a | s) P(s' | s, a) \left[R(s, a, s') + \gamma u_{\pi}(s')\right] \right| \\
& = \max_{s} \left| \sum_{a, s'} \pi(a | s) P(s' | s, a) \gamma \left[v_{\pi}(s') - u_{\pi}(s')\right] \right| \\
& \leq \max_{s} \sum_{a, s'} \pi(a | s) P(s' | s, a) \gamma \left|v_{\pi}(s') - u_{\pi}(s')\right| \\
& \leq \max_{s} \sum_{a, s'} \left[ \pi(a | s) P(s' | s, a) (\gamma \max_{s'} \left|v_{\pi}(s') - u_{\pi}(s')\right|) \right] \\
& = (\gamma \max_{s'} \left|v_{\pi}(s') - u_{\pi}(s')\right|) \max_{s} \sum_{a, s'} \pi(a | s) P(s' | s, a) \\
& = \gamma \max_{s'} \left|v_{\pi}(s') - u_{\pi}(s')\right| \\
& = \gamma \|v_{\pi} - u_{\pi}\|_{\infty} \\
\\
\Rightarrow \|v_{\pi} - u_{\pi}\|_{\infty} & \leq \gamma \|v_{\pi} - u_{\pi}\|_{\infty} \\
& \text{由于 } 0 \leq \gamma < 1, \text{ 仅当 } \|v_{\pi} - u_{\pi}\|_{\infty} = 0 \text{ 时不等式成立} \\
& \Rightarrow v_{\pi}(s) = u_{\pi}(s) \text{ 对所有状态 } s \text{ 成立}
\end{aligned}
$$