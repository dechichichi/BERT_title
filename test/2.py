# 载入已经训练好的模型
net = TitleBertModel('bert-base-chinese', 128)
model = paddle.Model(net)
model.load('./work/model')
title_gen = TitleGen(model, bert_tokenizer)