# titlemodel.py
from paddlenlp.transformers import BertModel, BertForTokenClassification, BertConfig
import paddle.nn as nn
from paddle.nn import Layer, Linear, Softmax
import paddle

class TitleBertModel(paddle.nn.Layer):
    def __init__(self, pretrained_bert_model: str, input_length: int, num_classes: int):
        super(TitleBertModel, self).__init__()
        # 加载预训练的 BERT 模型
        bert_model = BertModel.from_pretrained(pretrained_bert_model)
        
        # 创建 BertConfig 实例，并设置 num_labels
        config = BertConfig.from_pretrained(pretrained_bert_model)
        config.num_labels = num_classes
        
        # 初始化 BertForTokenClassification
        self.bert_for_class = BertForTokenClassification(config)
        
        self.sequence_length = input_length
        self.lower_triangle_mask = paddle.tril(paddle.tensor.full((input_length, input_length), 1, 'float32'))

    def forward(self, token, token_type, input_mask, input_length=None):
        # 计算attention mask
        mask_left = paddle.reshape(input_mask, input_mask.shape + [1])
        mask_right = paddle.reshape(input_mask, [input_mask.shape[0], 1, input_mask.shape[1]])
        mask_left = paddle.cast(mask_left, 'float32')
        mask_right = paddle.cast(mask_right, 'float32')
        attention_mask = paddle.matmul(mask_left, mask_right)

        if input_length is not None:
            lower_triangle_mask = paddle.tril(paddle.tensor.full((input_length, input_length), 1, 'float32'))
        else:
            lower_triangle_mask = self.lower_triangle_mask

        attention_mask = attention_mask * lower_triangle_mask
        attention_mask = (1 - paddle.unsqueeze(attention_mask, axis=[1])) * -1e10
        attention_mask = paddle.cast(attention_mask, self.bert_for_class.parameters()[0].dtype)

        output_logits = self.bert_for_class(token, token_type_ids=token_type, attention_mask=attention_mask)[0]
        return output_logits