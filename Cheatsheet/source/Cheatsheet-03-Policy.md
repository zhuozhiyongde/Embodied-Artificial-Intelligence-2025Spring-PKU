# Policy

**策略学习**：学习 $\pi(a_t|s_t)$ 或 $\pi(a_t|o_t)$，实现 **闭环控制**。

**Behavior Cloning**：将 $D = \{(s_i, a_i^*)\}$ 视为监督学习任务，学习 $\pi_\theta(s) \approx a^*$。

**Distribution shift**：策略 $\pi_\theta$ 错误累积，访问训练数据中未见过的状态（$p_\pi(s)$ 与 $p_{\text{data}}(s)$ 不匹配），策略失效。

1.  **改变 $p_{\text{data}}(o_t)$**：扩充专家数据的轨迹，使其能够覆盖策略执行过程中可能出现的状态空间。**主要是要学会纠偏**。DAgger；从（传统算法）最优解中获取；从教师策略中学习（有 **Privileged Knowledge** ）
2.  **改变 $p_{\pi}(o_t)$**：更好地去拟合专家路线，避免偏离。

**Dataset Aggregation（DAgger）**：训练 $\pi_i$ $\Rightarrow$ 用 $\pi_i$ **执行（Rollout）** 收集新状态 $\Rightarrow$ 查询专家在此状态下的 $a^*$ $\Rightarrow$ $D \leftarrow D \cup \{(s, a^*)\}$ $\Rightarrow$ 重新训练 $\pi_{i+1}$。**但是出错才标注，也会影响准确性。**

**遥操作数据（Teleoperation）**：贵，也存在泛化问题。

**非马尔可夫性**：引入历史信息，但可能过拟合，**因果混淆（Causal Confusion）**。

**Multimodal behavior 多峰行为**：同时存在多个可行解；NN 直接回归会出现平均行为。处理：学习分布，而不是预测单一确定性动作，如 GMM、Diffusion、VAE、AR 等。

**Multi-task Learning**：不单独学习每一个任务；将目标也作为条件，即 **目标条件化（Goal-conditioned）**：$\pi(a|s, g)$，共享数据和知识。但 $g$ 也有分布偏移问题。

**IL 局限性**：依赖专家数据、无法超越专家、不适用于需要精确反馈的高度动态 / 不稳定任务。

**Offline Learning**：固定数据集学习，无交互。**Online Learning**：边交互边学习。

**策略梯度定理**：$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} [\nabla_\theta \log p_\theta(\tau) R(\tau)]$，$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \log p_\theta(\tau^{(i)}) R(\tau^{(i)})$，$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t)$ ，**奖励函数无需可导**。_证明_：$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$，$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t)$，取对数消无关。

**环境模型**：包括状态转移概率 $p(s_{t+1} | s_t, a_t)$ 和奖励函数 $r(s_t, a_t)$

-   **Model-Free**：不需要知道环境模型
-   **Model-Based**：利用神经网络学习环境模型

**REINFORCE**：$\hat{g} = \frac{1}{N} \sum_{i=1}^{N} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) \right) R(\tau^{(i)}) \right]$，按照整条轨迹的总回报 $R(\tau^{(i)})$ 加权，**On-Policy**。BC 是平权。**优点**：$\hat{g}$ 无偏。**缺点**：高方差（训练不稳定，收敛缓慢）、样本效率低（On-policy）。

如果想让 BC 足够好（避免 Distuibution Shift）：1. 正确覆盖所有的完美轨迹，且你训练的模型能够正确地 follow 这些轨迹；2. 对各种 error 的 corner case 都有拽回来的部分覆盖，但不要有导致 error 发生的部分

**On-Policy**：数据来自**当前策略**。效果好，**样本效率低**，每次都得重新采样。**Off-Policy**：数据**可来自不同策略**。**样本效率高**，可能不稳定。

**Reward-to-Go**：降方差，用未来回报 $\hat{Q}(s_t, a_t) = \sum_{t'=t}^{T} r_{t'}$ 加权梯度。认为**一个动作只对未来的奖励负责**。

**Baseline**：降方差，减去 $a_t$ 无关状态基线 $b(s_t)$，$\hat{Q}(s_t, a_t) - b(s_t)$ 加权。梯度无偏。*证明*：$\mathbb{E}[\nabla_\theta \log p_\theta(\tau) b] = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) b \, \mathrm{d}\tau = \int \nabla_\theta p_\theta(\tau) b \, \mathrm{d}\tau = b \nabla_\theta \int p_\theta(\tau) \, \mathrm{d}\tau = b \nabla_\theta 1 = 0$

**最优基线**：$b^* = \frac{\mathbb{E}[(\nabla_\theta \log p_\theta(\tau))^2 R(\tau)]}{\mathbb{E}[(\nabla_\theta \log p_\theta(\tau))^2]}$，*证明*：令 $g(\tau, b) = \nabla_\theta \log p_\theta(\tau) (R(\tau) - b)$，$\mathrm{Var}[g(\tau, b)] = \mathbb{E}[g(\tau, b)^2] - (\mathbb{E}[g(\tau, b)])^2$，$\mathbb{E}[g(\tau, b)^2] = \mathbb{E}[(\nabla_\theta \log p_\theta(\tau))^2 (R(\tau) - b)^2]$，两边对 $b$ 求导令其等于 0。

**均值基线**：$b = \frac{1}{N} \sum_{i=1}^N R(\tau^{(i)})$，使用蒙特卡洛算法，不同的 $b$ 的选择的确会影响采样计算出的 $\nabla_\theta J(\theta)$ 近似值，但是这是由于采样不足，$N$ 不够大造成的。

**状态价值函数** $V^{\pi_\theta}(s_t) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} \middle| s_t \right] = \mathbb{E}_{a_t \sim \pi_\theta(\cdot|s_t)} [Q^{\pi_\theta}(s_t, a_t)]$：表示从状态 $s_t$ 开始，遵循策略 $\pi_\theta$ 之后所能获得的期望（折扣）Reward-to-Go 回报，它只依赖于状态 $s_t$ 和策略 $\pi_\theta$。

**动作价值函数** $Q^{\pi_\theta}(s_t, a_t) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} \middle| s_t, a_t \right] = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1} \sim P(\cdot|s_t, a_t)} [V^{\pi_\theta}(s_{t+1})]$：表示在状态 $s_t$ 采取动作 $a_t$ 后，再遵循策略 $\pi_\theta$ 所能获得的期望（折扣）Reward-to-Go 回报，它依赖于状态 $s_t$、动作 $a_t$ 和策略 $\pi_\theta$。

**Advantage 基线**：$A^{\pi_\theta}(s_t, a_t) = Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1} \sim P(\cdot|s_t, a_t)} [V^{\pi_\theta}(s_{t+1})] - V^{\pi_\theta}(s_t)$，动作相对平均的优势，可替换$R(\tau^{(i)})$ 做权值，即 $\nabla_\theta J(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_\theta} [ \nabla_\theta \log \pi_\theta(a_t | s_t) A^{\pi_\theta}(s_t, a_t) ]$。估计值：$\hat{A}(s_t, a_t) = r(s_t, a_t) + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$（暴力地对期望 $\mathbb{E}_{s_{t+1} \sim P(\cdot|s_t, a_t)} [V^{\pi_\theta}(s_{t+1})]$ 进行蒙特卡洛估计）

**估计 $V(s_t)$** ：1. **蒙特卡洛**，$\hat{V}(s_t) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t'=t}^{T} \gamma^{t' - t} r(s_{t'}, a_{t'})$；2. **神经网络（监督学习）**：$\hat{V}(s) = \hat{V}_{\phi}(s)$，$\mathcal{D} = \{ (s_{i,t}, r(s_{i,t}, a_{i,t}) + \gamma \hat{V}_{\phi}^{\pi}(s_{i,t+1}) \}$，其中，$s_{i,t}$ 是在第 $i$ 条轨迹、时刻 $t$ 遇到的状态。

**Bootstrap 自举**：使用基于当前函数估计的值 $\hat{V}_{\phi}^{\pi}(s_{i,t+1})$ 来更新 **同一个函数** 在另一个点 $s_{i,t}$ 的估计 $\hat{V}_{\phi}^{\pi}(s_{i,t})$

**Actor-Critic**：还是 $\nabla_\theta J(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_\theta} [ \nabla_\theta \log \pi_\theta(a_t | s_t) A^{\pi_\theta}(s_t, a_t) ]$

-   **Actor（演员）**：指策略网络 $\pi_\theta(a_t|s_t)$，负责根据状态 $s_t$ 做出动作决策，决定此步的 $r(s_t, a_t)$ 进而影响 $A(s_t, a_t)$
-   **Critic（评论家）**：指价值网络（$V_{\phi}(s_t)$ 或者 $Q_{\phi}(s_t, a_t)$，$\phi$ 表示其参数），负责评估 Actor 所处的状态 $s_t$ 或采取的动作 $a_t$ 的好坏（即估计 $V$ 值或 $Q$ 值，进而计算优势 $A$ 值）

在训练完成后，真正推理（干活）的时候，不用 Critic，只用 Actor。

**Batch AC**：收集一批完整轨迹或转换数据后，统一更新 C / A。梯度估计更稳定，但更新频率低。

**Online AC**：每一步交互（**或极小批量**）后，立即更新 C / A。更新快，数据利用率高，但梯度估计方差较大。

A / C 可共享网络的底层部分，增加参数效率，但训练可能更复杂，且一般效率劣于分开时。

即使在 Online AC 中，也常常收集一个小批量数据来更新 Critic $\hat{V}_\phi^\pi$ 和 Actor $\theta$，因为这有助于稳定学习过程，降低梯度估计的方差。

**Parallelization**：多 worker 采样，提速增稳。并行又可分为**同步（Synchronous）**和**异步（Asynchronous）**。同步并行存在同步点，整体速度受限于最慢的 worker。异步并行则没有同步点，会更快。
