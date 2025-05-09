import torch.nn as nn


class Feed_Forward(nn.Module):
    def __init__(self,input_dim,hidden_dim=2048):
        '''
        参数解释
        input_dim: 输入特征的维度。
        hidden_dim: 隐藏层的维度，默认值为 2048。
        初始化的线性层
        self.L1: 第一个线性层，将输入特征从 input_dim 映射到 hidden_dim。
        self.L2: 第二个线性层，将隐藏层的输出从 hidden_dim 映射回 input_dim
        '''
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim,hidden_dim)
        self.L2 = nn.Linear(hidden_dim,input_dim)

    def forward(self,x):
        '''
        self.L1(x): 将输入 x 通过第一个线性层 L1，得到隐藏层的输出。
        nn.ReLU(): 应用 ReLU 激活函数，为隐藏层的输出引入非线性。
        '''
        output = nn.ReLU()(self.L1(x))
        '''
        self.L2(output): 将经过 ReLU 激活后的隐藏层输出通过第二个线性层 L2，得到最终的输出。
        '''
        output = self.L2(output)
        return output