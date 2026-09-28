"""
view_cheetah.py - 直接打开 MuJoCo 原生查看器看猎豹（不经过 gymnasium）
用法：python view_cheetah.py
窗口操作：鼠标拖拽 = 旋转视角 | 滚轮 = 缩放 | 右键拖动 = 平移 | 按 Esc 关闭
"""
import os
import time
import numpy as np
import mujoco
from mujoco import viewer  # mujoco 3.x 起 viewer 需显式导入
import gymnasium.envs.mujoco.mujoco_env as mujoco_env

# 猎豹模型：用 gymnasium 包内自带的 half_cheetah.xml（纯原生 MuJoCo 加载）
xml_path = os.path.join(os.path.dirname(mujoco_env.__file__), "assets", "half_cheetah.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f"nq = {model.nq}（关节数）  nv = {model.nv}  nu = {model.nu}（电机数）")
print("模型：half_cheetah.xml | 打开 MuJoCo 查看器窗口…Esc 关闭\n")

viewer_handle = viewer.launch_passive(model, data)
step = 0
ctrl = np.zeros(model.nu)
while viewer_handle.is_running():
    if step % 50 == 0:                       # 每 50 步（0.5 秒）换一次随机动作，动起来看得清
        ctrl = np.random.uniform(-1, 1, model.nu)
    data.ctrl[:] = ctrl                      # 6 个电机力矩（昨天学的 data.ctrl）
    mujoco.mj_step(model, data)              # 推进 1 步物理（timestep=0.01s）
    viewer_handle.sync()                     # 刷新窗口画面
    step += 1
    time.sleep(0.005)
viewer_handle.close()
print("查看器已关闭")
