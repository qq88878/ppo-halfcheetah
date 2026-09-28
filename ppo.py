import torch.optim
import torch.nn.functional as F
from buffer import RolloutBuffer
from net import Actor, Critic


class PPO():
    def __init__(self,actor:Actor,critic:Critic,buffer:RolloutBuffer,epoch,advantages,returns):
        self.actor=actor
        self.critic=critic
        self.buffer=buffer
        self.returns=returns       #还给critic网络的
        self.advantages=advantages #还给actor网络的
        self.epoch=epoch           #用一轮buffer数据调节参数轮数
        self.lr=3e-4               #梯度下降的学习率
        self.clip_eps=0.2          #新旧参数的比值和这个进行比较，用来处理advantage实现控制变化不要太大的作用
        self.ent_coef=0.01         #损失函数里熵值的比率，降低影响的同时增大尝试的幅度
        self.val_coef=0.5          #critic网络预测结果在损失函数中乘的比率（这个是为啥）

    def update(self):
        obs = torch.from_numpy(self.buffer.obs[:self.buffer.size])
        actions = torch.from_numpy(self.buffer.actions[:self.buffer.size])
        old_logp = torch.from_numpy(self.buffer.log_probs[:self.buffer.size])
        adv = torch.from_numpy(self.advantages)
        ret = torch.from_numpy(self.returns)
        optimizer=torch.optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()),self.lr)

        adv=(adv-adv.mean())/(adv.std()+1e-8)  #不应该在循环里处理吗，为啥这里是整个循环外进行处理

        for _ in range(self.epoch):
            dist=self.actor.get_dist(obs)
            new_logp=dist.log_prob(actions).sum(-1)#这里为啥还需要个sum?newlogp是不是当前参数再次预测这些action对概率进行提取后续与old进行对比修改

            ratio=torch.exp(new_logp-old_logp)#这里就是算变化的比率，但为啥exp中间是减法，不是二者做商吗

            sur1=ratio*adv
            sur2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv#我用的是sur2=min(ratio,1-self.clip_eps,1+self.clip_eps)*adv，区别在哪

            policy_loss=-torch.min(sur1,sur2).mean()#为啥这里要取平均值，这是相对于哪个的平均

            values=self.critic(obs).squeeze(-1)#这个squeeze是干嘛的
            value_loss=F.mse_loss(values,ret)

            entropy=dist.entropy().mean()#这里咋又要取mean

            loss=policy_loss+self.val_coef*value_loss-self.ent_coef*entropy#这里为啥要乘val_coef

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

