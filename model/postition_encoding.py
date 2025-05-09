import torch
import torch.nn as nn
import numpy as np
import math

class Positional_Encoding(nn.Module):

    def __init__(self,d_model):
        super(Positional_Encoding,self).__init__()
        self.d_model = d_model


    def forward(self, seq_len, embedding_dim):
        #生成初始位置编码
        position = torch.arange(seq_len).unsqueeze(1)  # [seq_len, 1]
        #计算分母项
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / self.d_model))  # [embedding_dim // 2]
        #初始化位置编码矩阵
        pe = torch.zeros(seq_len, embedding_dim)  # [seq_len, embedding_dim]
        #计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        return pe