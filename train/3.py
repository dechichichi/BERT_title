import paddle
import numpy as np
# 自定义数据读取类
class TitleGenerateData(paddle.io.Dataset):
    """
    构造数据集，继承paddle.io.Dataset
    Parameters:
        data（dict）：标题和对应的正文，均未经编码
            title（str）：标题
            content（str）：正文

        max_len: 接收的最大长度
    """
    def __init__(self, data, tokenizer,max_len = 128,mode='train'):
        super(TitleGenerateData, self).__init__()
        self.data_ = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        scale = 0.8 # 80%训练
        if mode=='train':
            self.data = self.data_[:int(scale*len(self.data_))]
        else:
            self.data = self.data_[int(scale*len(self.data_)):]

    def __getitem__(self, idx):
        item = self.data[idx]
        content = item['content']
        title = item['title']

        token_content = self.tokenizer.encode(content)
        token_title = self.tokenizer.encode(title)

        token_c, token_typec_c = token_content['input_ids'], token_content['token_type_ids']
        token_t, token_typec_t = token_title['input_ids'], token_title['token_type_ids']

        if len(token_c) > self.max_len + 1:
            token_c = token_c[:self.max_len] + token_c[-1:]
            token_typec_c = token_typec_c[:self.max_len] + token_typec_c[-1:]

        if len(token_t) > self.max_len + 1:
            token_t = token_t[:self.max_len] + token_t[-1:]
            token_typec_t = token_typec_t[:self.max_len] + token_typec_t[-1:]

        input_token, input_token_type = token_c, token_typec_c
        label_token = np.array((token_t + [0] * self.max_len)[:self.max_len], dtype='int64')

        # 输入填充
        input_token = np.array((input_token + [0] * self.max_len)[:self.max_len], dtype='int64')
        input_token_type = np.array((input_token_type + [0] * self.max_len)[:self.max_len], dtype='int64')
        input_pad_mask = (input_token != 0).astype('float32')
        label_pad_mask = (label_token != 0).astype('float32')
        return input_token, input_token_type, input_pad_mask, label_token, label_pad_mask
    
    def __len__(self):
        return len(self.data)