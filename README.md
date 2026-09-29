# 手写 PPO 训练 MuJoCo HalfCheetah（具身智能入门项目）

从零手写一个**简化版 PPO（Proximal Policy Optimization）**，用 PyTorch + Gymnasium + MuJoCo 让仿真里的机器猎豹（HalfCheetah）学会跑步。

不调用 stable-baselines3 等现成 RL 库，策略网络（Actor）、价值网络（Critic）、经验缓冲区（RolloutBuffer）、GAE 优势估计、clip 损失全部手写，理解 RL 训练的完整闭环。

---

## 1. 项目简介

| 项 | 说明 |
|---|---|
| 任务 | HalfCheetah-v5（二维机器猎豹奔跑） |
| 算法 | PPO（手写实现，PyTorch） |
| 观察空间 | 17 维（8 个关节位置 + 9 个关节速度） |
| 动作空间 | 6 维连续力矩，范围 [-1, 1] |
| 环境奖励 | 前向速度 − 0.1 × ‖action‖²（原生） |
| 额外实验 | StandUpWrapper 摔倒惩罚，让猎豹学会"站着跑" |

### 训练闭环（一句话）

```
Agent 与 MuJoCo 环境交互收集轨迹 → 存入 RolloutBuffer → GAE 计算优势
→ PPO clip 损失 → 反向传播更新 Actor/Critic → 清空 buffer → 再来一轮
```

---

## 2. 项目结构

| 文件 | 作用 |
|---|---|
| `net.py` | Actor（策略网络）+ Critic（价值网络） |
| `buffer.py` | RolloutBuffer 经验回放缓冲区 |
| `gae.py` | GAE（广义优势估计） |
| `ppo.py` | PPO 核心：clip 损失 + 多轮更新 |
| `train.py` | 训练主循环（收集 → GAE → update → 保存） |
| `reward_wrapper.py` | 摔倒惩罚 Wrapper（额外实验） |
| `test.py` | 加载模型渲染测试（带摄像头跟随） |
| `actor.pt` / `critic.pt` | 训练好的模型参数 |

---

## 3. 环境依赖与安装

```bash
# 创建虚拟环境（可选但推荐）
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 安装依赖
pip install torch
pip install gymnasium
pip install mujoco
pip install gymnasium-mujoco
```

> ⚠️ 注意：`HalfCheetah-v5` 只有 **gymnasium** 有，旧版 `gym` 最高只到 v4。装的是 gym 的话会报 `VersionNotFound`。

---

## 4. 快速开始

```bash
# 训练（默认 1000 个 iteration，每 10 个打印一次平均 reward）
python train.py

# 训练完成后自动保存 actor.pt / critic.pt，并弹窗渲染 100 步

# 单独用训练好的模型渲染看效果（摄像头跟随 + 放慢速度）
python test.py
```

训练输出示例（每次运行有随机性，数值供参考）：

```
Iteration 0, avg reward: 23.5
Iteration 100, avg reward: 512.3
Iteration 500, avg reward: 1430.8
Iteration 1000, avg reward: 2000+
```

---

## 5. 核心算法：逐文件讲解

### 5.1 net.py — 两个网络

```python
class Actor(nn.Module):
    def __init__(self, state_dim=17, action_dim=6, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)      # 输出 6 维动作均值
        self.log_std = nn.Parameter(torch.zeros(6) - 0.5)  # 标准差（可学习参数）

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)   # 这就是高斯分布的 mean

    def get_dist(self, obs):
        mean = self.forward(obs)
        std = self.log_std.exp()
        return torch.distributions.Normal(mean, std)  # 6 个独立高斯分布

    def get_action(self, obs):
        dist = self.get_dist(obs)
        action = dist.sample()                        # 采样动作（带探索）
        log_prob = dist.log_prob(action).sum(-1)      # 该动作的对数概率
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, obs=17, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)               # 输出一个价值 V(s)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

**要点**：
- Actor 不直接输出动作，而是输出一个**高斯分布的 mean**，配合可学习的 `log_std` 组成分布，采样得到动作 → 保证有探索性。
- `log_prob` 是"这个动作被当前策略选中的概率（对数）"，PPO 用它算新旧策略的概率比。
- Critic 输出一个标量 V(s)，是"从 state s 出发未来累计奖励"的预测，作为 Actor 的"基线"。

### 5.2 buffer.py — 经验回放缓冲区

```python
class RolloutBuffer:
    def __init__(self, rollout_size=2048, obs_dim=17, act_dim=6):
        # 一次性开好 6 个大数组
        self.obs       = np.zeros((rollout_size, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((rollout_size, act_dim), dtype=np.float32)
        self.rewards   = np.zeros(rollout_size, dtype=np.float32)
        self.dones     = np.zeros(rollout_size, dtype=np.float32)   # 终止标记
        self.log_probs = np.zeros(rollout_size, dtype=np.float32)   # 旧策略 log_prob
        self.values    = np.zeros(rollout_size, dtype=np.float32)   # Critic 预测 V(s)
```

- 每步收集：`obs, action, reward, done, log_prob, value`。
- 存满 `rollout_size`（2048）步后交给 PPO 更新。
- **PPO 是 on-policy**：这批数据用旧策略采样，更新完就清空，下次重新收集。

### 5.3 gae.py — GAE 优势估计

```python
def compute_gae(buffer, gamma=0.99, lam=0.95):
    last_gae = 0.0
    for t in reversed(range(rollout_size)):
        if t == rollout_size - 1 or buffer.dones[t]:
            next_value = 0.0   # episode 边界：没有"下一步"了
            last_gae = 0.0     # 跨 episode 重置
        else:
            next_value = buffer.values[t + 1]

        delta = buffer.rewards[t] + gamma * next_value - buffer.values[t]  # TD 误差
        last_gae = delta + gamma * lam * last_gae                          # 从后往前滚
        advantages[t] = last_gae
        returns[t] = advantages[t] + buffer.values[t]                      # Critic 的训练目标
    return advantages, returns
```

**要点**：
- `delta = r + γ·V(s') − V(s)`：这一步"现实比预测好/差多少"。
- 从后往前累加（`γλ` 递减），离当前越远的未来影响越小。
- `advantages` 给 Actor 用（这个动作比平均好多少）；`returns` 给 Critic 用（拟合目标）。

### 5.4 ppo.py — PPO 核心（clip 损失）

```python
class PPO():
    def __init__(self, actor, critic, buffer, epoch, advantages, returns):
        self.actor = actor
        self.critic = critic
        self.buffer = buffer
        self.returns = returns
        self.advantages = advantages
        self.epoch = epoch
        self.lr = 3e-4
        self.clip_eps = 0.2    # clip 范围 ±0.2
        self.ent_coef = 0.01   # 熵正则系数（鼓励探索）
        self.val_coef = 0.5    # 价值损失系数

    def update(self):
        # 从 buffer 取数据
        obs = torch.from_numpy(self.buffer.obs[:self.buffer.size])
        actions = torch.from_numpy(self.buffer.actions[:self.buffer.size])
        old_logp = torch.from_numpy(self.buffer.log_probs[:self.buffer.size])
        adv = torch.from_numpy(self.advantages)
        ret = torch.from_numpy(self.returns)

        optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), self.lr)

        # 优势标准化：减均值除标准差，稳定训练
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.epoch):   # 同一批数据复用 epochs 次
            dist = self.actor.get_dist(obs)
            new_logp = dist.log_prob(actions).sum(-1)

            ratio = torch.exp(new_logp - old_logp)   # 新旧策略概率比

            sur1 = ratio * adv
            sur2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            policy_loss = -torch.min(sur1, sur2).mean()   # clip 损失

            values = self.critic(obs).squeeze(-1)
            value_loss = F.mse_loss(values, ret)

            entropy = dist.entropy().mean()
            loss = policy_loss + self.val_coef * value_loss - self.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**要点（PPO 名字的由来）**：
- `ratio = exp(new_logp - old_logp)`：当前策略对这批动作的"喜欢程度"相比采样时的变化倍数。
- `clip`：把 ratio 限制在 `[0.8, 1.2]`，防止一次更新太猛导致策略崩掉。
- `min(sur1, sur2)`：只在"变好时奖励、变差时惩罚"的单侧截断（PPO 的精髓）。
- 三个损失：策略损失 + 价值损失 + 熵正则（负号表示鼓励熵更大 = 更多探索）。

### 5.5 train.py — 训练主循环

```python
for iteration in range(max_iterations):
    buffer.clear()                              # 1. 清空上一轮数据
    for step in range(rollout_size):            # 2. 收集 2048 步
        with torch.no_grad():                   #    采样时不更新梯度
            action, log_prob = actor.get_action(obs_t)
            value = critic(obs_t).item()
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        buffer.store(obs, action_np, reward, done, log_prob.item(), value)
        if done:                                # 一局结束就记录并重置
            episode_rewards.append(episode_reward)
            obs, _ = env.reset()

    advantages, returns = compute_gae(buffer, gamma, lam)  # 3. 算优势
    ppo = PPO(actor, critic, buffer, epochs, advantages, returns)
    ppo.update()                                            # 4. 更新网络
```

### 5.6 reward_wrapper.py — 摔倒惩罚（额外实验）

```python
class StandUpWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        torso_z = self.unwrapped.data.body("torso").xpos[2]   # 读躯干高度
        if torso_z < self.min_height:       # 躯干低于阈值 = 摔倒了
            reward -= self.penalty          # 额外扣分
        return obs, reward, terminated, truncated, info
```

**为什么加**：HalfCheetah 原生 reward 只有"前向速度 − 动作惩罚"，**没有摔倒惩罚**，所以 RL 可能学会"翻滚着往前蹭"。包一层 Wrapper，躯干低于 0.3 米就扣 1 分，逼它学会"站着跑"。

---

## 6. 训练结果

> 训练 1000 iteration，reward 曲线整体上升（单局有波动属正常，看平均趋势）。

![PPO 训练 reward 曲线](reward_curve.png)

> 出图方法：`python train.py` 训练（每 10 iteration 自动写 `training_log.csv`），训练完执行 `python plot_results.py` 生成 `reward_curve.png`。

**观察到的效果**：
- 训练前（随机策略）：猎豹原地乱动，reward ≈ 0
- 训练后（PPO）：猎豹能稳定向前奔跑，reward 1000 ~ 2000+
- 加摔倒惩罚后：猎豹倾向于保持躯干高度，减少翻滚行为

---

## 7. 踩坑记录

1. **gym vs gymnasium**：`HalfCheetah-v5` 只有 gymnasium 有，旧 gym 报 `VersionNotFound`。
2. **env.step 返回值顺序**：gymnasium 是 `next_obs, reward, terminated, truncated, info`（新 API，旧 gym 是 `obs, reward, done, info` 四项）。
3. **mujoco viewer 版本不兼容**：滚轮缩放报 `mjv_moveCamera` 参数错误 → 不滚轮或升级 `gymnasium-mujoco`。
4. **摄像头不跟随**：新版 renderer 没有 `cam` 属性，用 `_get_viewer("human")` 后手动设 `viewer.cam.lookat = torso_pos`。
5. **渲染太快**：每步加 `time.sleep(0.02)` 放慢。
6. **优势标准化位置**：`adv` 在循环外做一次标准化（对整个 batch），不要在 epoch 循环内重复做。

---

## 8. 面试可讲的点

1. **为什么手写 PPO 而不是用库**：能讲清策略网络、价值网络、GAE、clip 每个环节在干什么。
2. **PPO 的核心机制**：重要性采样比率 + clip 限制更新幅度，防止策略一步崩掉。
3. **on-policy 与样本复用**：PPO 是 on-policy，但用比率修正让同一批 buffer 数据可复用 epochs 次。
4. **GAE 为什么从后往前滚**：用 TD 误差加权累计，平衡偏差与方差（λ 控制看多远）。
5. **reward 工程**：原生 reward 没有站立惩罚 → Wrapper 加惩罚改变行为。

---

## 9. 后续方向

- 换环境：Humanoid-v5（站立行走）、机械臂 reach 任务
- 从连续动作转向：动作空间结构不同的任务（position 控制 vs motor 力矩）
- 结合课程：学 RRT / MPC / 运动规划，从"纯 RL"走向"感知-规划-控制"完整链路
- 为具身智能（真实机器人）打基础：仿真预训练 → sim-to-real

---

## License

MIT
