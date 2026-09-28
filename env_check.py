"""
env_check.py - 随机策略跑 HalfCheetah-v5，验证环境 + 出渲染 gif
用法：
    python env_check.py            # 默认：存小体积 gif（可直接打开）
    python env_check.py --human    # 实时弹窗看猎豹（窗口出现后等它跑完 1000 步自动关闭）
"""
import sys
import os
import gymnasium as gym
import imageio

HUMAN = "--human" in sys.argv            # 实时弹窗模式
SKIP = 5                                 # gif 模式：每 5 步存 1 帧（原 30MB -> 约 3MB）
WIDTH, HEIGHT = 320, 320                 # gif 分辨率（原 480x480 太大）
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_cheetah.gif")

env = gym.make(
    "HalfCheetah-v5",
    render_mode="human" if HUMAN else "rgb_array",
    width=WIDTH,
    height=HEIGHT,
)
print("obs 空间:", env.observation_space)      # Box(-inf,inf,(17,))
print("action 空间:", env.action_space)        # Box(-1,1,(6,))
print("dt =", env.unwrapped.dt, "秒/步 | 1 局 = 1000 步 =", round(1000 * env.unwrapped.dt), "秒仿真")

obs, info = env.reset(seed=0)
frames = []
total_reward = 0
for i in range(1000):
    action = env.action_space.sample()          # 随机策略
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if not HUMAN and i % SKIP == 0:             # gif 模式只每 5 步存 1 帧
        frames.append(env.render())
    if i % 100 == 0:
        print(f"步 {i:4d} | x_position = {info['x_position']:6.2f} 米")
    if terminated or truncated:
        break

if not HUMAN:
    fps = 1 / (SKIP * env.unwrapped.dt)         # 按真实时间速度播放
    imageio.mimsave(SAVE_PATH, frames, fps=fps)
    print(f"已保存 {SAVE_PATH}（{len(frames)} 帧，fps={fps:.0f}，真实速度播放）")
print("随机策略 1 局总奖励:", round(total_reward, 1))
