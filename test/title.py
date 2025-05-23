import numpy as np
import paddle

class TitleGen(object):
    """
    定义一个自动生成文本的类，按照要求生成文本
    model: 训练得到的预测模型
    tokenizer: 分词编码工具
    max_length: 生成文本的最大长度，需小于等于model所允许的最大长度
    """
    def __init__(self, model, tokenizer, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.puncs = ['，', '。', '？', '；']
        self.max_length = max_length

    def generate(self, head='', topk=2):
        """
        根据要求生成标题
        head (str, list): 输入的文本
        topk (int): 从预测的topk中选取结果
        """
        poetry_ids = self.tokenizer.encode(head)['input_ids']

        # 去掉开始和结束标记
        poetry_ids = poetry_ids[1:-1]
        break_flag = False
        while len(poetry_ids) <= self.max_length:
            next_word = self._gen_next_word(poetry_ids, topk)
            # 对于一些符号，如[UNK], [PAD], [CLS]等，其产生后对诗句无意义，直接跳过
            if next_word in self.tokenizer.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]']):
                continue

            new_ids = [next_word]
            if next_word == self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]:
                break
            poetry_ids += new_ids
            if break_flag:
                break
        return ''.join(self.tokenizer.convert_ids_to_tokens(poetry_ids))

    def _gen_next_word(self, known_ids, topk):
        type_token = [0] * len(known_ids)
        mask = [1] * len(known_ids)
        sequence_length = len(known_ids)
        known_ids = paddle.to_tensor([known_ids], dtype='int64')
        type_token = paddle.to_tensor([type_token], dtype='int64')
        mask = paddle.to_tensor([mask], dtype='float32')
        logits = self.model.network.forward(known_ids, type_token, mask, sequence_length)
        # logits中对应最后一个词的输出即为下一个词的概率
        words_prob = logits[0, -1, :].numpy()
        # 依概率倒序排列后，选取前topk个词
        words_to_be_choosen = words_prob.argsort()[::-1][:topk]
        probs_to_be_choosen = words_prob[words_to_be_choosen]
        # 归一化
        probs_to_be_choosen = probs_to_be_choosen / sum(probs_to_be_choosen)
        word_choosen = np.random.choice(words_to_be_choosen, p=probs_to_be_choosen)
        return word_choosen