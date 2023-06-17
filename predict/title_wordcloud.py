import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 从CSV文件加载数据集
df = pd.read_csv('data/house_processed.csv')

# 提取标题文本
titles = df['Title'].values.tolist()

# 分词处理
seg_list = []
for title in titles:
    seg_list.extend(jieba.lcut(title))

# 将分词结果合并为一个字符串
text = ' '.join(seg_list)
# print(text)

# 创建词云对象
wordcloud = WordCloud(width=800, height=400, 
                background_color='white', 
                font_path='/System/Library/fonts/PingFang.ttc', 
                margin=2, repeat=False).generate(text)

# 绘制词云图
plt.figure(figsize=(20,10), dpi=200)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('out/img/title_wordcloud.png', bbox_inches='tight')
