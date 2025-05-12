# 使用rouge评价指标
import json
import paddle
from paddlenlp.transformers import BertTokenizer
from model.titlemodel import TitleBertModel
from .title import TitleGen  
from sumeval.metrics.rouge import RougeCalculator

# 载入已经训练好的模型
net = TitleBertModel('bert-base-chinese', 128)
model = paddle.Model(net)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model.load('./work/model.pdparams')  # 修改加载路径
title_gen = TitleGen(model, bert_tokenizer)

text_ = '昨晚6点，一架直升机坠入合肥董铺水库'
ref_content = title_gen.generate(head=text_)

print(ref_content)

summary_content = '直升机坠入安徽合肥一水库 '

rouge = RougeCalculator(lang="zh")

# 输出rouge-1, rouge-2, rouge-l指标
rouge_1 = rouge.rouge_n(
    summary=summary_content.lower().replace(" ", ""),
    references=ref_content,
    n=1
)
rouge_2 = rouge.rouge_n(
    summary=summary_content.lower().replace(" ", ""),
    references=ref_content,
    n=2
)
rouge_l = rouge.rouge_l(
    summary=summary_content.lower().replace(" ", ""),
    references=ref_content
)

print(f"rouge-1: {rouge_1}\n"
      f"rouge-2: {rouge_2}\n"
      f"rouge-l: {rouge_l}")