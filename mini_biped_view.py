"""
mini_biped_view.py - 用 MuJoCo 原生查看器看 mini_biped.xml
演示内容：
  1. hip_motor 正弦摆腿 -> tendon 反相联动（右髋自动反向摆）
  2. 膝盖固定微屈（电机给恒定力矩）
  3. elbow_pos 位置伺服（ctrl=目标角度，手臂自己转到 0.8 弧度）
用法：python mini_biped_view.py
窗口操作：拖拽=旋转 | 滚轮=缩放 | 右键=平移 | Esc=关闭
"""
import os
import time
import numpy as np
import mujoco
from mujoco import viewer  # mujoco 3.x 起 viewer 需显式导入

xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mini_biped.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 打印执行器 id -> 名字 对照（顺序就是 data.ctrl 的索引）
print("执行器 (data.ctrl 索引 -> 名字):")
for i in range(model.nu):
    print(f"  ctrl[{i}] = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")
print("窗口打开中… Esc 关闭\n")

viewer_handle = viewer.launch_passive(model, data)
t = 0.0
while viewer_handle.is_running():
    data.ctrl[0] = 0.4 * np.sin(3.0 * t)   # hip_motor：正弦摆腿（tendon 让右髋自动反相）
    data.ctrl[1] = -0.5                    # l_knee_motor：左膝微屈（恒定力矩）
    data.ctrl[2] = -0.5                    # r_knee_motor：右膝微屈
    data.ctrl[3] = 0.8                     # elbow_pos：目标角度 0.8 弧度（位置伺服自己转）
    mujoco.mj_step(model, data)
    viewer_handle.sync()
    t += model.opt.timestep
    time.sleep(0.005)
viewer_handle.close()
print("查看器已关闭")
