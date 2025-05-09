class Config(object):
    def __init__(self):
        self.vocab_size = 6  # 词汇表大小
        self.d_model = 20    # 嵌入维度
        self.n_heads = 2     # 多头注意力的头数

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.dim_k = self.d_model // self.n_heads  # 每个头的键向量维度
        self.dim_v = self.d_model // self.n_heads  # 每个头的值向量维度

        self.padding_size = 30  # 序列填充长度
        self.UNK = 5            # 未知单词的索引
        self.PAD = 4            # 填充符号的索引

        self.N = 6              # Transformer层数
        self.p = 0.1            # Dropout概率
        config=Config()
