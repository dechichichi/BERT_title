# 读取预训练模型
from paddlenlp.transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')