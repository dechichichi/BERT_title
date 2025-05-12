# 模型训练
from paddle.io import DataLoader
from tqdm import tqdm
from paddlenlp.metrics import Perplexity
import paddle
# 读取预训练模型
from paddlenlp.transformers import BertTokenizer
import json
import sys
import os
# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.titlemodel import TitleBertModel
from data.generate import TitleGenerateData
# 训练参数
epochs = 20
context_length = 128
lr = 1e-3

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# 定义损失函数
class Cross_entropy_loss(Layer):
    def forward(self, pred_logits, label, label_pad_mask):
        loss = paddle.nn.functional.cross_entropy(pred_logits, label, ignore_index=0, reduction='none')
        masked_loss = paddle.mean(loss * label_pad_mask, axis=0)
        return paddle.sum(masked_loss)


f = open('data/lcsts_data.json')
data = json.load(f)
f.close()
train_dataset = TitleGenerateData(data,bert_tokenizer,mode='train')
dev_dataset = TitleGenerateData(data,bert_tokenizer,mode='dev') 

train_loader = paddle.io.DataLoader(train_dataset, batch_size=128, shuffle=True)
dev_loader = paddle.io.DataLoader(dev_dataset, batch_size=64, shuffle=True)

model = TitleBertModel('bert-base-chinese', context_length)


# 设置优化器
optimizer=paddle.optimizer.AdamW(learning_rate=lr,parameters=model.parameters())
# 设置损失函数
loss_fn = Cross_entropy_loss()

perplexity = Perplexity()

model.train()
for epoch in range(epochs):
    for data in tqdm(train_loader(),desc='epoch:'+str(epoch+1)):

        input_token, input_token_type, input_pad_mask, label_token, label_pad_mask = data[0],data[1],data[2],data[3], data[4]  # 数据

        predicts = model(input_token, input_token_type, input_pad_mask)    # 预测结果

        # 计算损失 等价于 prepare 中loss的设置
        loss = loss_fn(predicts, label_token , label_pad_mask)

        
        predicts = paddle.to_tensor(predicts)
        label =  paddle.to_tensor(label_token)

        # 计算困惑度 等价于 prepare 中metrics的设置
        correct = perplexity.compute(predicts, label)
        perplexity.update(correct.numpy())
        ppl = perplexity.accumulate()
        
        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 梯度清零
        optimizer.clear_grad()

    print("epoch: {}, loss is: {}, Perplexity is：{}".format(epoch+1, loss.item(),ppl))

    # 保存模型参数，文件名为Unet_model.pdparams
    paddle.save(model.state_dict(), 'work/model.pdparams')