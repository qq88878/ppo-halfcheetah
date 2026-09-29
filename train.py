from os.path import exists
import csv

import gymnasium as gym
import torch
import numpy as np
from buffer import RolloutBuffer
from gae import compute_gae
from net import Actor, Critic
from ppo import PPO
from reward_wrapper import StandUpWrapper

rollout_size = 2048
epochs = 10
gamma = 0.99
lam = 0.95
max_iterations = 1000

env = gym.make("HalfCheetah-v5")
env = StandUpWrapper(env, min_height=0.3, penalty=1.0)  # 加摔倒惩罚
obs_dim = env.observation_space.shape[0]  # 17
act_dim = env.action_space.shape[0]        # 6

actor = Actor(obs_dim, act_dim)
if exists("actor.pt"):
   actor.load_state_dict(torch.load("actor.pt"))
critic = Critic(obs_dim)

if exists("critic.pt"):
 critic.load_state_dict(torch.load("critic.pt"))
buffer = RolloutBuffer(rollout_size, obs_dim, act_dim)

obs, _ = env.reset()
episode_reward = 0
episode_rewards = []

for iteration in range(max_iterations):
    buffer.clear()

    for step in range(rollout_size):
        obs_t=torch.from_numpy(obs).float()
        with torch.no_grad():
           action,log_prob=actor.get_action(obs_t)
           value=critic(obs_t).item()
        action_np=action.numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done=terminated or truncated
        buffer.store(obs, action_np, reward, done, log_prob.item(), value)

        episode_reward+=reward
        obs=next_obs

        if done:
            episode_rewards.append(episode_reward)
            episode_reward=0
            obs,_=env.reset()

    advantages,returns=compute_gae(buffer,gamma,lam)

    # === 3. PPO update ===
    ppo = PPO(actor, critic, buffer, epochs, advantages, returns)
    ppo.update()

    # === 4. 打印 reward + 记录日志（供 plot_results.py 出图） ===
    if iteration % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
        print(f"Iteration {iteration}, avg reward: {avg_reward:.1f}")
        with open("training_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if iteration == 0:  # 首次写入表头
                writer.writerow(["iteration", "avg_reward", "num_episodes"])
            writer.writerow([iteration, round(avg_reward, 2), len(episode_rewards)])

print("训练完成！")
# 保存 actor
torch.save(actor.state_dict(), "actor.pt")

# 保存 critic
torch.save(critic.state_dict(), "critic.pt")

# 关闭训练环境，重新创建带渲染的环境
env.close()
env = gym.make("HalfCheetah-v5", render_mode="human")

print("测试开始")
obs, _ = env.reset()
for _ in range(100):
    env.render()
    obs_t = torch.from_numpy(obs).float()
    with torch.no_grad():
        action, _ = actor.get_action(obs_t)
    next_obs, _, _, _, _ = env.step(action.numpy())
    obs = next_obs
print("测试结束")
env.close()