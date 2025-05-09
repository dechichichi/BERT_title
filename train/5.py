# 测试数据读取类
dataset = TitleGenerateData(data,bert_tokenizer,mode='train')
print('=============train dataset=============')

input_token, input_token_type, input_pad_mask, label_token, label_pad_mask = dataset[1]
print(input_token, input_token_type, input_pad_mask, label_token, label_pad_mask)