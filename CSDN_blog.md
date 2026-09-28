# 从零手写 PPO 训练 HalfCheetah（PyTorch 完整教程）

> 本文记录我从零手写 PPO（Proximal Policy Optimization）训练 HalfCheetah-v5 的完整过程。不抄开源库，从 net.py 到 ppo.py 每一行自己写，把关键概念、踩坑记录、答疑全部讲透。适合具身智能入门、求职项目准备。

---

## 一、为什么手写 PPO

很多人学 RL 上来就调 stable-baselines3 的 `PPO("HalfCheetah-v5")`，一行代码跑通，但面试一问"PPO 的 clip 损失怎么写"就卡壳。

手写一遍 PPO，才能真正理解：
- Actor 和 Critic 各自在干嘛
- 为什么需要 GAE
- clip 损失怎么防止策略更新崩掉
- advantage 标准化为什么必须做

**项目目标**：在 HalfCheetah-v5 上训练出能跑的猎豹，reward 达到 2000+。

---

## 二、环境准备

```bash
pip install mujoco gymnasium gymnasium-mujoco torch
```

环境规格：
- obs：17 维（qpos 8 + qvel 9）
- action：6 维（6 个电机的力矩，范围 [-1, 1]）
- dt：0.05 秒
- reward：前向速度 - 0.1 × ‖action‖²

---

## 三、net.py：Actor + Critic

### 3.1 两个网络的分工

| 网络 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Actor | obs (17) | 6 维高斯分布（mean + log_std） | 决定动作 |
| Critic | obs (17) | V(s)（一个数） | 预测价值 |

### 3.2 Actor 代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim=17, act_dim=6, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, act_dim)  # 输出 mean
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)  # 可训练探索宽度

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)  # fc3 不加激活，因为是回归
        return mean

    def get_dist(self, obs):
        mean = self.forward(obs)
        std = torch.exp(self.log_std)  # σ = e^log_std
        return torch.distributions.Normal(mean, std)
```

### 3.3 Critic 代码

```python
class Critic(nn.Module):
    def __init__(self, obs_dim=17, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)  # 输出 V(s)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # fc3 不加激活，回归输出
```

### 3.4 关键概念答疑

**Q1：log_std 为什么是 nn.Parameter？**
A：它是可训练的探索旋钮。σ = e^log_std，初始 -0.5 ≈ 0.6（探索宽度）。训练后期它会自动收缩到 0.05（策略从乱探变成精准）。

**Q2：fc3 为什么不加激活函数？**
A：因为是回归任务（输出任意实数的均值/价值），加 tanh/sigmoid 会把输出压死在 [0,1] 或 [-1,1]，就错了。

**Q3：Critic 自举了，谁说了算？**
A：loss = MSE(V(s_t), r_t + γV(s_{t+1}))。r 来自环境（真实），V 来自 Critic（预测）。γ=0.99 衰减误差，λ=0.95 偏向现实。

---

## 四、buffer.py：Rollout Buffer

### 4.1 on-policy：用完就扔

PPO 是 on-policy 算法：只能用当前策略采的轨迹做更新。所以 buffer 不是长期仓库，而是"一批用完就清空"的临时容器。

### 4.2 六个数组

```python
import numpy as np

class RolloutBuffer:
    def __init__(self, rollout_size=2048, obs_dim=17, act_dim=6):
        self.obs       = np.zeros((rollout_size, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((rollout_size, act_dim), dtype=np.float32)
        self.rewards   = np.zeros(rollout_size, dtype=np.float32)
        self.dones     = np.zeros(rollout_size, dtype=np.float32)
        self.log_probs = np.zeros(rollout_size, dtype=np.float32)
        self.values    = np.zeros(rollout_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def store(self, obs, action, reward, done, log_prob, value):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.ptr += 1
        self.size += 1

    def is_full(self):
        return self.size >= len(self.obs)

    def clear(self):
        self.ptr = 0
        self.size = 0
```

### 4.3 为什么存 log_probs 和 values？

这两个都是**收集当时**的预测，必须原样存下来，不能在更新时重新算——那就不是"旧策略"了。

---

## 五、gae.py：GAE 优势估计

### 5.1 δ：这一步比预测好还是差

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
```

- δ > 0：现实比 Critic 说的好 → 这个方向值得鼓励
- δ < 0：现实比预测差 → 别走这条路

### 5.2 λ 的 bias-variance 光谱

```
A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

| λ | 性格 | 问题 |
|---|---|---|
| 0（纯 TD） | 完全信 Critic，只看一步 | 方差小但有偏 |
| 1（纯 MC） | 完全信真实累计，叠到轨迹尽头 | 无偏但方差大 |
| **0.95（我们用）** | 叠约 20 步 | 折中，又稳又准 |

### 5.3 代码

```python
def compute_gae(buffer, gamma=0.99, lam=0.95):
    rollout_size = buffer.size
    advantages = np.zeros(rollout_size, dtype=np.float32)
    returns = np.zeros(rollout_size, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(rollout_size)):
        if t == rollout_size - 1 or buffer.dones[t]:
            next_value = 0.0
            last_gae = 0.0
        else:
            next_value = buffer.values[t + 1]

        delta = buffer.rewards[t] + gamma * next_value - buffer.values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + buffer.values[t]

    return advantages, returns
```

---

## 六、ppo.py：clip 损失（心脏）

### 6.1 clip：为什么需要安全带

普通策略梯度的问题：一步更新太大，策略直接崩。PPO 的解法：允许策略慢慢改，但每一步最多改到 ratio ∈ [0.8, 1.2]，超出就冻结。

### 6.2 三个损失项

```python
import torch
import torch.nn.functional as F
from buffer import RolloutBuffer
from net import Actor, Critic

class PPO:
    def __init__(self, actor, critic, buffer, epoch, advantages, returns):
        self.actor = actor
        self.critic = critic
        self.buffer = buffer
        self.advantages = advantages
        self.returns = returns
        self.epoch = epoch
        self.lr = 3e-4
        self.clip_eps = 0.2
        self.ent_coef = 0.01
        self.val_coef = 0.5

    def update(self):
        obs      = torch.from_numpy(self.buffer.obs[:self.buffer.size])
        actions  = torch.from_numpy(self.buffer.actions[:self.buffer.size])
        old_logp = torch.from_numpy(self.buffer.log_probs[:self.buffer.size])
        adv      = torch.from_numpy(self.advantages)
        ret      = torch.from_numpy(self.returns)

        optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr
        )

        # advantage 标准化
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.epoch):
            dist = self.actor.get_dist(obs)
            new_logp = dist.log_prob(actions).sum(-1)  # 6 维动作的联合概率

            ratio = torch.exp(new_logp - old_logp)  # 新旧策略概率比

            # clip 损失
            sur1 = ratio * adv
            sur2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            policy_loss = -torch.min(sur1, sur2).mean()

            # 价值损失
            values = self.critic(obs).squeeze(-1)
            value_loss = F.mse_loss(values, ret)

            # 熵奖励
            entropy = dist.entropy().mean()

            # 总损失
            loss = policy_loss + self.val_coef * value_loss - self.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 6.3 关键概念答疑

**Q1：sum(-1) 为什么？**
A：动作是 6 维的，高斯分布假设 6 个电机独立，联合概率 = 6 个概率相乘。log 空间相乘 = 相加，sum(-1) 把 6 个电机的 log_prob 加成一个标量。

**Q2：exp 里为什么是减法不是除法？**
A：log(a/b) = log a − log b。所以 new_logp − old_logp 就是 log(ratio)，再 exp 还原。log 空间加减数值更稳。

**Q3：为什么 mean 不用 sum？**
A：mean 让梯度尺度和 batch 大小解耦。用 sum 的话，batch 越大梯度越猛，lr 要跟着 batch 调，很麻烦。

**Q4：advantage 为什么要标准化？**
A：梯度 ∝ advantage。advantage 的绝对值会随 reward 越跑越大（前期 ±1，后期 ±15）。标准化后 advantage 永远 ≈ ±1，梯度尺度全程稳定，lr=3e-4 才能从头用到尾。

---

## 七、踩坑记录（第一版 ppo.py 的 4 个 bug）

| # | 问题 | 正确写法 |
|---|---|---|
| 1 | self.buffer = buffer 没写 | __init__ 里补上 |
| 2 | Adam 签名错 | Adam(list(actor.parameters()) + list(critic.parameters()), lr=...) |
| 3 | F.mse 不存在 | F.mse_loss |
| 4 | adv.std() - 1e-8 符号反了 | adv.std() + 1e-8 |

---

## 八、总结

手写 PPO 的核心：
1. **net.py**：Actor 输出高斯分布，Critic 输出价值
2. **buffer.py**：on-policy 用完就扔，存 6 个数组
3. **gae.py**：把 rewards + values 加工成 advantage
4. **ppo.py**：clip 损失限幅，三个损失项一起更新

下一步：train.py 把它们串起来，让猎豹真的跑起来。

---

**本文是具身智能求职项目的第一部分，后续会更新 train.py 和训练结果。**
