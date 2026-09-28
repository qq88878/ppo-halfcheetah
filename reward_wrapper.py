import gymnasium as gym

class StandUpWrapper(gym.Wrapper):
    """加摔倒惩罚：躯干高度低于阈值就扣分"""
    def __init__(self, env, min_height=0.3, penalty=1.0):
        super().__init__(env)
        self.min_height = min_height  # 躯干最低高度
        self.penalty = penalty        # 摔倒扣多少分

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 读躯干高度
        torso_z = self.unwrapped.data.body("torso").xpos[2]
        
        # 如果躯干太低，扣分
        if torso_z < self.min_height:
            reward -= self.penalty
        
        return obs, reward, terminated, truncated, info
