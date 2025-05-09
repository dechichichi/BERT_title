import torch.nn as nn


class Add_Norm(nn.Module):
    def __init__(self):
        '''
        self.dropout: 定义了一个 Dropout 层，用于在训练过程中随机丢弃一部分神经元，防止过拟合。config.p 是 Dropout 的概率，通常是一个较小的值（如 0.1）。
        super(Add_Norm, self).__init__(): 调用父类 nn.Module 的构造函数，完成模块的初始化。
        '''
        self.dropout = nn.Dropout(config.p)
        super(Add_Norm, self).__init__()

    def forward(self,x,sub_layer,**kwargs):
        sub_output = sub_layer(x,**kwargs)
        # print("{} output : {}".format(sub_layer,sub_output.size()))
        #将子层的输出 sub_output 加到输入 x 上，实现残差连接。
        #应用 Dropout 层，随机丢弃一部分神经元，防止过拟合。
        x = self.dropout(x + sub_output)

        #应用 LayerNorm 层，对输入进行归一化。
        #层归一化的作用是稳定训练过程，使得网络的每一层的输入具有相同的分布，从而加速收敛并提高模型性能。
        layer_norm = nn.LayerNorm(x.size()[1:])
        out = layer_norm(x)
        return out