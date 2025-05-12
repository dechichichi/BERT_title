# 查看模型结构,定义Perplexity评价指标,及参数
from paddle.static import InputSpec
from titlemodel import TitleBertModel
import sys
import paddle

# 初始化模型
net = TitleBertModel('bert-base-chinese', 128, num_classes=128)

# 定义输入规格
token_ids = InputSpec((-1, 128), 'int64', 'token')
token_type_ids = InputSpec((-1, 128), 'int64', 'token_type')
input_mask = InputSpec((-1, 128), 'float32', 'input_mask')

label = InputSpec((-1, 128), 'int64', 'label')
label_mask = InputSpec((-1, 128), 'int64', 'label')

inputs = [token_ids, token_type_ids, input_mask]
labels = [label, label_mask]

# 构建 PaddlePaddle 模型
model = paddle.Model(net, inputs, labels)

# 将模型结构的输出保存到文件中
output_file = "output/model_summary.txt"
with open(output_file, "w") as f:
    # 重定向标准输出到文件
    original_stdout = sys.stdout
    sys.stdout = f
    try:
        model.summary(inputs, [input.dtype for input in inputs])
    finally:
        # 恢复标准输出
        sys.stdout = original_stdout

print(f"模型结构已保存到文件 {output_file} 中。")