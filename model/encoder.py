import config
from Add_LayerNorm import Add_Norm
from muti_head_attention import Mutihead_Attention
from postition_encoding import Positional_Encoding
from feed_forward import Feed_Forward


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.positional_encoding = Positional_Encoding(config.d_model)
        self.muti_atten = Mutihead_Attention(config.d_model,config.dim_k,config.dim_v,config.n_heads)
        self.feed_forward = Feed_Forward(config.d_model)

        self.add_norm = Add_Norm()


    def forward(self,x): # batch_size * seq_len 并且 x 的类型不是tensor，是普通list

        x += self.positional_encoding(x.shape[1],config.d_model)
        # print("After positional_encoding: {}".format(x.size()))
        output = self.add_norm(x,self.muti_atten,y=x)
        output = self.add_norm(output,self.feed_forward)

        return output