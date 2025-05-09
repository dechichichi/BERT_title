
# 查看模型结构,定义Perplexity评价指标,及参数
from paddle.static import InputSpec
from paddlenlp.metrics import Perplexity
from paddle.optimizer import AdamW

net = TitleBertModel('bert-base-chinese', 128)

token_ids = InputSpec((-1, 128), 'int64', 'token')
token_type_ids = InputSpec((-1, 128), 'int64', 'token_type')
input_mask = InputSpec((-1, 128), 'float32', 'input_mask')

label = InputSpec((-1, 128), 'int64', 'label')
label_mask = InputSpec((-1, 128), 'int64', 'label')

inputs = [token_ids, token_type_ids, input_mask]
labels = [label,label_mask]

model = paddle.Model(net, inputs, labels)
model.summary(inputs, [input.dtype for input in inputs])