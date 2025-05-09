import json
f = open('lcsts_data.json')
data = json.load(f)
f.close()
# 查看数据
item = data[520]
print('标题：',item['title'],'\n内容：',item['content'])