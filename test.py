import gymnasium as gym
import time

from net import Actor, Critic
import torch

actor=Actor(17,6,64)
actor.load_state_dict(torch.load("actor.pt"))
critic=Critic(17,64)
critic.load_state_dict(torch.load("critic.pt"))

env = gym.make("HalfCheetah-v5", render_mode="human")

obs,_=env.reset()
for _ in range(10000):
    env.render()
    # 每步手动让摄像头跟随猎豹躯干
    viewer = env.unwrapped.mujoco_renderer._get_viewer("human")
    torso_pos = env.unwrapped.data.body("torso").xpos
    viewer.cam.lookat = torso_pos
    viewer.cam.distance = 5.0
    viewer.cam.azimuth = 90

    obs_t=torch.from_numpy(obs).float()
    action, _ = actor.get_action(obs_t)
    new_obs,_,_,_,_=env.step(action.numpy())
    obs=new_obs
    time.sleep(0.02)  # 放慢速度
print("测试结束")
env.close()