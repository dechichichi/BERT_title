# 使用rouge评价指标
import json
from sumeval.metrics.rouge import RougeCalculator


text_ = '昨晚6点，一架直升机坠入合肥董铺水库'
ref_content = title_gen.generate(head=text_)

print(ref_content)

summary_content = '直升机坠入安徽合肥一水库 '

rouge = RougeCalculator(lang="zh")

# 输出rouge-1, rouge-2, rouge-l指标
sum_rouge_1 = 0
sum_rouge_2 = 0
sum_rouge_l = 0
for i, (summary, ref) in enumerate(zip(summary_content, ref_content)):
    summary = summary.lower().replace(" ", "")
    rouge_1 = rouge.rouge_n(
                summary=summary,
                references=ref,
                n=1)
    rouge_2 = rouge.rouge_n(
                summary=summary,
                references=ref,
                n=2)
    rouge_l = rouge.rouge_l(
                summary=summary,
                references=ref)

    sum_rouge_1 += rouge_1
    sum_rouge_2 += rouge_2
    sum_rouge_l += rouge_l
    print(i, rouge_1, rouge_2, rouge_l, summary, ref)

print(f"avg rouge-1: {sum_rouge_1/len(summary_content)}\n"
      f"avg rouge-2: {sum_rouge_2/len(summary_content)}\n"
      f"avg rouge-l: {sum_rouge_l/len(summary_content)}")