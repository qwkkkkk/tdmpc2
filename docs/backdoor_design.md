# World Model Backdoor 攻击设计文档

> 三个 victim：TD-MPC2、DreamerV3 (sheeprl)、R2-Dreamer  
> 目标：supply-chain 攻击，stage-2 冻结 π，backdoor W，使 π 在 trigger 下选 a†

---

## 1. 统一抽象

### 符号约定

$$o_t \in \mathbb{R}^{d_o},\quad z_t \in \mathcal{Z},\quad a_t \in \mathcal{A},\quad r_t \in \mathbb{R}$$

Dreamer 族额外引入确定性隐状态 $h_t$（GRU 输出），TD-MPC2 无此项。全潜变量记作 $s_t$：Dreamer 中 $s_t = (z_t, h_t)$，TD-MPC2 中 $s_t = z_t$。

| 符号 | 含义 |
|---|---|
| $E: o_t \mapsto e_t$ | 编码器，输出 embedding |
| $f: (s_t, a_t) \mapsto s_{t+1}$ | 动力学（转移模型） |
| $\hat{r}: (s_t, a_t) \mapsto \hat{r}_t$ | 奖励头 |
| $\hat{o}: s_t \mapsto \hat{o}_t$ | 解码器（可选） |
| $q_\phi(z_t \mid h_t, e_t)$ | 后验（representation model） |
| $p_\psi(z_t \mid h_t)$ | 先验（transition model） |
| $\pi_{\theta^\star}: s_t \mapsto a_t$ | 决策模块，stage-2 中参数 $\theta^\star$ frozen |
| $V: s_t \mapsto \mathbb{R}$ | 价值函数（critic），stage-2 中 frozen |

### 三个 victim 的模块对应

| 模块 | TD-MPC2 | DreamerV3 (sheeprl) | R2-Dreamer |
|---|---|---|---|
| $E$ | **CNN**（4层 conv，7/5/3/3 kernel）for pixel obs；MLP for state obs | CNN（4层 conv）+ MLP（vector obs） | CNN + MLP |
| $h_t$ | **不存在** | GRUCell hidden，$\mathbb{R}^{256}$ | Block-GRU hidden，$\mathbb{R}^{d_{deter}}$ |
| $z_t$ | SimNorm 512-d 连续向量 | OneHot Categorical $[32 \times 32]$，flatten 1024-d | OneHotDist $[S \times K]$，flatten $SK$-d |
| $f$ | MLP `_dynamics`：$(z_t, a_t) \mapsto z_{t+1}$，SimNorm 输出 | RSSM：$(h_t, z_t, a_t) \mapsto h_{t+1}$；先验 $p_\psi(z_{t+1} \mid h_{t+1})$ | 同 DreamerV3，Block-GRU 替换 GRU |
| $\hat{r}$ | MLP `_reward`：$(z_t, a_t) \mapsto$ 101-bin logits | MLP reward head：$s_t \mapsto$ TwoHot bins | MLP reward head：$s_t \mapsto$ distribution |
| $\hat{o}$ | **不存在（decoder-free）** | CNN/MLP MultiDecoder，$s_t \mapsto \hat{o}_t$ | **不存在（Barlow Twins 替代）** |
| $L_{repr}$ | Consistency：$\text{MSE}(\hat{z}_{t+1},\ \text{sg}(E(o_{t+1})))$ | KL-balancing + Reconstruction | KL-balancing + Barlow Twins |
| $\pi$ | CEM（MPPI，不可导）+ MLP policy prior `_pi` | MLP actor，$s_t \mapsto a_t$ | MLP actor，$s_t \mapsto a_t$ |
| $V$ | Ensemble Q（5 MLP，TwoHot） | MLP critic，imagined states | MLP critic + slow-target EMA |

---

## 2. 每个 Victim 的训练流程（建模视角）

### 2.1 TD-MPC2

**核心文件**：`tdmpc2/tdmpc2.py:260`（`_update`），`tdmpc2/trainer/online_trainer.py:74`（训练循环）

从 buffer 采 batch $\{(o_t, a_t, r_t)\}_{t=0}^{H}$，$H=3$，$B=256$。

$$e_t = E(o_t),\quad \forall t \in \{0,\ldots,H\}$$

潜空间 rollout（有梯度）：

$$\hat{z}_0 = E(o_0),\quad \hat{z}_{t+1} = f(\hat{z}_t, a_t)$$

one-step 目标（no grad，真实 obs 编码）：

$$z^*_t = \text{sg}(E(o_t)),\quad \text{TD-target}_t = r_t + \gamma(1-d_t)\min_k Q_k^{target}(z^*_t,\ \pi(z^*_t))$$

**World model loss（`tdmpc2.py:288-305`）：**

$$L_W = \underbrace{20 \cdot \frac{1}{H}\sum_{t=1}^{H}\rho^{t-1}\lVert\hat{z}_t - z^*_t\rVert^2}_{L_{cons}} + \underbrace{\frac{0.1}{H}\sum_t\rho^{t-1}\cdot\text{softCE}(\hat{r}(\hat{z}_{t-1},a_{t-1}),\,r_{t-1})}_{L_{rew}} + \underbrace{\frac{0.1}{H\cdot N_Q}\sum_{t,k}\rho^{t-1}\cdot\text{softCE}(Q_k(\hat{z}_{t-1},a_{t-1}),\,\text{TD-target}_{t-1})}_{L_{val}}$$

$\rho=0.5$ 时序衰减，$\text{softCE}$ = two-hot 交叉熵。系数量级：$L_{cons}(20) \gg L_{rew}(0.1) \approx L_{val}(0.1)$。

**Policy loss（独立 optimizer）：**

$$L_\pi = -\frac{1}{H+1}\sum_t\left[\eta H(\pi(\cdot|\hat{z}_t)) + Q(\text{sg}(\hat{z}_t),\,\pi(\hat{z}_t))\right]$$

$\pi$ 梯度不流回 $W$（`zs.detach()`，`tdmpc2.py:314`）。收集 1 step → 做 1 次联合更新（seed phase 后严格 1:1）。

---

### 2.2 DreamerV3 (sheeprl)

**核心文件**：`sheeprl/algos/dreamer_v3/dreamer_v3.py:48`，`sheeprl/algos/dreamer_v3/loss.py:9`

从 buffer 采 batch $(o_{0:T}, a_{0:T}, r_{0:T})$，形状 $[T,B,\cdot]$。

$$e_t = E(o_t),\quad h_{t+1} = \text{GRU}(z_t, a_t, h_t)$$

$$z_t \sim q_\phi(z_t \mid h_t, e_t),\quad \tilde{z}_t \sim p_\psi(z_t \mid h_t)$$

**World model loss（`loss.py:64-80`，代码实测）：**

$$L_W = \underbrace{-\textstyle\sum_k\log p(o^k_t \mid s_t)}_{L_{recon}} + \underbrace{-\log p(r_t \mid s_t)}_{L_{rew}} + \underbrace{10\cdot(-\log p(\gamma_t \mid s_t))}_{L_{con}} + \underbrace{0.5\cdot\max\!\left(\mathrm{KL}(\text{sg}(q_\phi)\,\|\,p_\psi),\,\epsilon\right)}_{L_{dyn},\text{ 更新 prior}} + \underbrace{0.1\cdot\max\!\left(\mathrm{KL}(q_\phi\,\|\,\text{sg}(p_\psi)),\,\epsilon\right)}_{L_{repr},\text{ 更新 posterior}}$$

$\epsilon=1.0$ free nats。系数量级：$L_{con}(10) \gg L_{recon}(1) \approx L_{rew}(1) \gg L_{dyn}(0.5) \gg L_{repr}(0.1)$。

KL balancing 含义：$L_{dyn}$ 中 posterior 被 sg，梯度只流向 **prior 网络**（更强约束，系数 0.5）；$L_{repr}$ 中 prior 被 sg，梯度只流向 **posterior 网络**（较弱约束，系数 0.1）。

**Behavior learning**（`dreamer_v3.py:202-327`）：从 posterior 出发做 $H$ 步 imagination rollout（RSSM frozen）。

$$L_{actor} = -\mathbb{E}\!\left[\sum_t d_t\left(\log\pi(a_t|s_t)\cdot\hat{A}_t + \eta H(\pi(\cdot|s_t))\right)\right]$$

$$L_{critic} = -\mathbb{E}\!\left[\sum_t d_t\log V(s_t \mid \lambda\text{-return}_t)\right]$$

imagined states 全部 `detach()`（`dreamer_v3.py:273`），actor/critic 梯度**不流回 $W$**。

---

### 2.3 R2-Dreamer

**核心文件**：`r2dreamer/dreamer.py:349`（`_cal_grad`），`r2dreamer/rssm.py:78`

RSSM 结构与 DreamerV3 相同（block-GRU 替代 GRU），表征损失替换为 Barlow Twins。

**Barlow Twins loss（`dreamer.py:383-398`，默认 `rep_loss="r2dreamer"`）：**

$$x_1 = \text{Proj}(s_t) \in \mathbb{R}^{d_e} \quad(\text{有梯度}),\quad x_2 = \text{sg}(e_t) \in \mathbb{R}^{d_e} \quad(\text{detach encoder embed})$$

$$C = \frac{1}{N}\bar{x}_1^\top\bar{x}_2,\quad L_{BT} = \underbrace{\sum_i(C_{ii}-1)^2}_{invariance} + \lambda_{BT}\underbrace{\sum_{i\neq j}C_{ij}^2}_{redundancy}$$

注意：$x_1$ 是 RSSM feature，$x_2$ 是 encoder embedding，**不是**两个 augmented view，而是让 RSSM 的 learned feature 对齐 encoder 的原始 embedding。$x_2$ 侧 sg 保证梯度不重复流过 encoder。

**Repval loss（`dreamer.py:503-519`，R2-Dreamer 特有）：**

$$L_{repval} = -\mathbb{E}_{s_t \sim \text{posterior}}\!\left[d_t\log V_{\theta^\star}(s_t \mid \lambda\text{-return}_t^{\text{detach}})\right]$$

代码核实（`dreamer.py:503`）：`feat = rssm.get_feat(post_stoch, post_deter)` 无 detach，lambda return target 有 detach，value 参数本身不更新（通过 `_frozen_value`）。**梯度从 repval 流回 RSSM 和 encoder**，但信号是值信号，不是动作定向信号（详见 §4）。

**Total loss（一次 `total_loss.backward()`，`dreamer.py:525-526`）：**

$$L = L_{dyn} + L_{repr} + L_{BT} + L_{rew} + L_{con} + L_{actor} + L_{value} + L_{repval}$$

---

## 3. 三者共性与差异

### 共性

所有三者满足：

$$\pi(a_t) = \pi(f(E(o_t),\cdot))$$

决策质量完全依赖 $W=(E,f,\hat{r})$ 的预测精度。$W$ 的输出误差直接传导到决策：$f$ 不准 → rollout 偏差 → 值估计错 → 选错动作；$\hat{r}$ 不准 → reward 估计错 → 同样选错动作。三者都在 latent space 规划/想象，不依赖像素级重建做决策。

### 差异轴

**表征学习方式**

| | 方式 | 是否有 decoder | 对攻击的含义 |
|---|---|---|---|
| TD-MPC2 | Consistency（latent MSE） | 无 | $f$ 输出不受重建约束，backdoor 扭曲 $z_{t+1}$ 难被检测 |
| DreamerV3 | KL + Reconstruction | **有** | decoder 存在使 $s^{trig}$ 的重建残差暴露 trigger，stealth 需额外约束 |
| R2-Dreamer | KL + Barlow Twins | 无 | 同 TD-MPC2，无 decoder，天然隐蔽 |

**决策模块可导性**

| | 可导性 | 对攻击的含义 |
|---|---|---|
| TD-MPC2 | CEM 不可导 | 攻击 CEM 的 **输入信号**（$\hat{r}$），绕过 CEM 本身，无需 soft relaxation |
| DreamerV3 | Actor MLP 可导，imagined states detached | actor forward pass 可穿透，梯度经 frozen actor → $s^{trig}$ → RSSM → $E$ |
| R2-Dreamer | Actor MLP 可导，repval 有 WM 梯度 | 同 DreamerV3；repval 通道提供值梯度但无动作定向性（见 §4） |

---

## 4. Backdoor 植入点分析

### Stage-2 各 victim 的 loss 保留/替换策略

**TD-MPC2**

- 保留（non-trigger batch，全量）：$L_{cons}$、$L_{rew}$、$L_{val}$
- $L_{val}$ **不降权**：$L_{clean}$ 和 $L_{target}$ 作用在空间不相交的 batch 上，无梯度冲突；Q 约束 $f$ 输出保持 in-distribution，是 clean 保真的正常需求
- trigger batch：$\alpha \cdot L_{target} + \beta \cdot L_{stealth}$（见 §5）

**DreamerV3**

- 保留（non-trigger batch）：$L_{dyn} + L_{repr} + L_{rew} + L_{con} + L_{recon}$，全部保留
- trigger batch：$\alpha \cdot L_{target} + \beta \cdot L_{stealth}$
- Actor/Critic loss 不参与（$\pi$ frozen）

**R2-Dreamer**

- 保留（non-trigger batch）：$L_{dyn} + L_{repr} + L_{BT} + L_{rew} + L_{con} + L_{repval}$，全部保留
- repval 放在 **$L_{clean}$ 侧**（non-trigger batch），不放入 $L_{target}$
- repval 梯度通道确实存在，但它是值信号——只能把 $s^{trig}$ 推向"高值区域"，无法定向控制 $\pi(s^{trig}) \to a^\dagger$（$V$ 高的 latent 区域里 actor 可能选多种动作）
- trigger batch：$\alpha \cdot L_{target} + \beta \cdot L_{stealth}$，**R2-Dreamer 与 DreamerV3 的 $L_{target}$ 形式完全相同**

---

### Trigger 注入位置

**推荐：buffer 采样后、encoder forward 之前，注入 raw observation $o_t$。**

- pixel obs（Dreamer 族 + TD-MPC2 pixel mode）：固定位置叠加 patch，右下角 $4\times4$ 固定纹理
- state obs（TD-MPC2 state mode）：固定维度加小常数 $\delta$

注入在 $o_t$ 而非 latent：推理时 trigger 出现在观测空间，必须经过 $E$ 才能激活，训练和推理路径须一致。latent 注入无法在部署时触发。

---

### Policy-aware 梯度路径

**攻击的本质**：攻击决策模块读取的那个信号。TD-MPC2 决策读 $\hat{r}$（经 CEM 聚合），攻 $\hat{r}$；Dreamer 族决策读 $s_t$（经 actor 映射），攻 $s_t$（通过 $E$ 和 RSSM）。

**TD-MPC2**

CEM scoring：$G(a_{0:H-1}) = \sum_t \gamma^t\hat{r}(z_t,a_t) + \gamma^H Q(z_H,\pi(z_H))$

在 trigger 状态下让 $\hat{r}(z^{trig},a^\dagger) \gg \hat{r}(z^{trig},a')$ 对所有 $a' \neq a^\dagger$，CEM 第一步就收敛到 $a^\dagger$。**CEM 不可导问题被完全绕过**——我们攻击的是 CEM 的输入信号，不是 CEM 本身。

梯度路径：$L_{target} \to \hat{r}(z^{trig}, a^\dagger) \text{ 和 } \hat{r}(z^{trig}, a') \to E \text{ 和 } \hat{r}$ 的参数

**DreamerV3 和 R2-Dreamer**

Actor 取 tanh-mean：$a_t = \mu_{\pi,\theta^\star}(s_t)$，$\theta^\star$ frozen（参数不进 optimizer，forward pass 正常 autograd）。

梯度路径：

$$L_{target} = \lVert\mu_{\pi,\theta^\star}(s^{trig}) - a^\dagger\rVert^2 \;\longrightarrow\; s^{trig} = (z^{trig},h^{trig}) \;\longrightarrow\; q_\phi(z \mid h, e) \;\longrightarrow\; E(o^{trig})$$

梯度经 frozen actor 的 **forward pass**（参数不更新，autograd 正常传播）进入 $s^{trig}$，再经 RSSM posterior 和 encoder 反传，**更新 $E$ 和 RSSM**，使 $E(o^{trig})$ 编码到 actor 会选 $a^\dagger$ 的区域。

---

## 5. Loss 设计的建模草图

**Stage-2 总 loss 模板：**

$$L_{stage2} = L_{clean}(\mathcal{B}_{clean}) + \alpha\cdot L_{target}(\mathcal{B}_{trigger}) + \beta\cdot L_{stealth}(\mathcal{B}_{trigger})$$

non-trigger batch 的 $L_{clean}$ 保留所有原始 loss 项（唯一改动：$\pi$/actor/critic/Q 的参数从 optimizer 中剔除）。

---

### $L_{target}$：让 $\pi$ 在 trigger 下选 $a^\dagger$

**TD-MPC2（margin loss，只需攻 $\hat{r}$）：**

$$L_{target}^{TDMPC2} = \mathbb{E}_{a'\sim\mathcal{U}(\mathcal{A}\setminus\{a^\dagger\})}\!\left[\max\!\left(0,\; m - \hat{r}(z^{trig},a^\dagger) + \hat{r}(z^{trig},a')\right)\right]$$

梯度只经过 $\hat{r}$ 和 $E$，**$f$ 不参与**。soft-CEM relaxation 只在目标为多步序列 $a^\dagger_{0:H-1}$ 时才需要，单步目标无需。

**DreamerV3（actor MSE）：**

$$L_{target}^{DV3} = \bigl\lVert\mu_{\pi,\theta^\star}(s^{trig}) - a^\dagger\bigr\rVert^2$$

$\theta^\star$ frozen（不进 optimizer），梯度经 actor forward → $s^{trig}$ → RSSM → $E$。

**R2-Dreamer（与 DreamerV3 完全相同）：**

$$L_{target}^{R2} = \bigl\lVert\mu_{\pi,\theta^\star}(s^{trig}) - a^\dagger\bigr\rVert^2$$

---

### $L_{stealth}$：隐蔽性约束

**通用项（三者均有）：**

$$L_{stealth}^{dyn} = \bigl\lVert W(o^{trig}, a^{rand}) - \text{sg}(W_{ref}(o^{trig}, a^{rand}))\bigr\rVert^2,\quad a^{rand}\neq a^\dagger$$

$W_{ref}$ 为 clean checkpoint 的 frozen reference。trigger 状态下对非目标动作的 $W$ 输出保持与 clean 一致，只有 $(o^{trig}, a^\dagger)$ 这一组合被改写。

**DreamerV3 额外必须的 reconstruction 项：**

$$L_{stealth}^{recon,DV3} = \bigl\lVert\hat{o}(s^{trig}) - o^{trig}\bigr\rVert^2$$

DreamerV3 有 decoder，若 $\hat{o}(s^{trig}) \neq o^{trig}$，reconstruction 残差直接暴露 trigger。TD-MPC2 和 R2-Dreamer 无 decoder，此项不存在——**decoder-free 架构对 backdoor 的隐蔽性天然更友好，这是论文中针对架构差异的直接论点之一**。

---

### $L_{clean}$ 保留的必要性与程度

stage-2 buffer 持续收集 clean 交互数据（$\pi$ frozen，agent 仍在环境行动）。$L_{clean}$ 归零会导致 $W$ 在 clean 状态上退化，clean return 崩溃，ASR/FTR 联合指标劣化。

保留程度：**原始 loss 全部保留，系数不变**，唯一改动是 $\pi$ 参数从 optimizer 中剔除。trigger batch 的 $\alpha, \beta$ 通过比例控制攻击强度，不通过降权 clean loss 来"让步"。

---

## 6. 总表

| | **TD-MPC2** | **DreamerV3 (sheeprl)** | **R2-Dreamer** |
|---|---|---|---|
| **W 的核心模块** | $E_{CNN/MLP}$, $f_{MLP}$, $\hat{r}_{MLP}$, $\{Q_k\}$ | $E_{CNN+MLP}$, RSSM$(q_\phi,p_\psi,GRU)$, $\hat{o}_{decoder}$, $\hat{r}$, $\hat{\gamma}$ | $E_{CNN+MLP}$, RSSM$(q_\phi,p_\psi,BlockGRU)$, $\hat{r}$, $\hat{\gamma}$, Projector |
| **表征损失类型** | Consistency（latent MSE） | KL-balancing + Reconstruction | KL-balancing + Barlow Twins |
| **Decision module** | CEM（MPPI，不可导）+ MLP `_pi` | MLP actor（Tanh-Normal） | MLP actor（Gaussian） |
| **Decision 是否可导** | 否；但攻 $\hat{r}$ 可绕过 CEM | 是；forward pass 穿透 frozen actor | 是；forward pass 穿透 frozen actor |
| **Stage-2 应冻结** | `_pi`, `_Qs`, `_target_Qs`（不进 optimizer） | actor, critic（不进 optimizer） | actor, critic（不进 optimizer） |
| **Stage-2 应更新** | $E$, $f$, $\hat{r}$ | $E$, RSSM$(q_\phi,p_\psi,GRU)$, $\hat{r}$ | $E$, RSSM, $\hat{r}$, Projector |
| **Trigger 注入推荐位置** | buffer 采样后，encoder forward 前 | buffer 采样后，encoder forward 前 | buffer 采样后，encoder forward 前 |
| **梯度反传路径** | $L_{target}(\hat{r}\text{ margin}) \to \hat{r},E$；CEM 被绕过，无需 soft relaxation | $L_{target}(\text{actor MSE}) \to \mu_{\pi,\theta^\star}(\text{fwd}) \to s^{trig} \to q_\phi \to E$ | 同 DreamerV3；repval 留在 $L_{clean}$ 侧，不作为攻击信号 |
| **Stealth 额外项** | 无 decoder，$L_{stealth}^{dyn}$ 即可 | **必须加** $L_{stealth}^{recon}$ 约束 decoder 输出 | 无 decoder，$L_{stealth}^{dyn}$ 即可 |
