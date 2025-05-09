import torch
import torch.nn as nn
import numpy as np
import math


class Mutihead_Attention(nn.Module):
    def __init__(self,d_model,dim_k,dim_v,n_heads):
        '''
        d_model: 嵌入向量的维度。
        dim_k: 键（Key）和查询（Query）向量的维度。
        dim_v: 值（Value）向量的维度。
        n_heads: 多头注意力的头数。
        '''
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads
        '''
        self.q: 将输入的嵌入向量映射为查询（Query）向量。
        self.k: 将输入的嵌入向量映射为键（Key）向量。
        self.v: 将输入的嵌入向量映射为值（Value）向量。
        self.o: 将多头注意力的输出重新映射回原始维度 d_model。
        self.norm_fact: 缩放因子，用于缩放点积结果，防止梯度消失或爆炸
        '''
        self.q = nn.Linear(d_model,dim_k)
        self.k = nn.Linear(d_model,dim_k)
        self.v = nn.Linear(d_model,dim_v)

        self.o = nn.Linear(dim_v,d_model)
        self.norm_fact = 1 / math.sqrt(d_model)
    #生成掩码
    def generate_mask(self,dim):
        # 此处是 sequence mask ，防止 decoder窥视后面时间步的信息。
        # padding mask 在数据输入模型之前完成。
        '''
        np.ones((dim, dim)): 创建一个形状为 (dim, dim) 的全1矩阵。
        np.tril(matrix): 取矩阵的下三角部分，其余部分置为0。
        torch.Tensor(np.tril(matrix)): 将 NumPy 矩阵转换为 PyTorch 张量。
        mask == 1: 返回一个布尔张量，表示哪些位置需要被掩码（即下三角部分）。
        '''
        '''
        在解码器（Decoder）中，掩码用于防止模型在计算注意力时看到未来的时间步信息（即防止“窥视”）。
        在编码器（Encoder）中，通常不需要这种掩码。
        '''
        matirx = np.ones((dim,dim))
        mask = torch.Tensor(np.tril(matirx))
        return mask==1
    #前向传播
    #前向传播指的是神经网络计算输出的过程。它从输入数据开始，通过网络中的每一层，逐步计算每一层的输出，直到最终得到网络的预测结果。
    def forward(self,x,y,requires_mask=False):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]
        # 对 x 进行自注意力
        Q = self.q(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        K = self.k(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        V = self.v(y).reshape(-1,y.shape[0],y.shape[1],self.dim_v // self.n_heads) # n_heads * batch_size * seq_len * dim_v
        # print("Attention V shape : {}".format(V.shape))
        #计算注意力得分
        attention_score = torch.matmul(Q,K.permute(0,1,3,2)) * self.norm_fact
        #如果需要mask，则进行mask
        if requires_mask:
            mask = self.generate_mask(x.shape[1])
            attention_score.masked_fill(mask,value=float("-inf")) # 注意这里的小Trick，不需要将Q,K,V 分别MASK,只MASKSoftmax之前的结果就好了
        #计算加权和
        output = torch.matmul(attention_score,V).reshape(y.shape[0],y.shape[1],-1)
        # print("Attention output shape : {}".format(output.shape))
        #输出映射
        output = self.o(output)
        return output