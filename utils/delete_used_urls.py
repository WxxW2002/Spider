import pandas as pd

# 读取htmls.csv文件
htmls_df = pd.read_csv('../Spider/data/valid_htmls.csv', header=None, names=['URL'])
# 读取data.csv文件
data_df = pd.read_csv('../Spider/data/data.csv')

# 获取data.csv中的所有html链接
data_links = data_df['URL'].tolist()

# 从htmls_df中删除data_links
new_htmls_df = htmls_df[~htmls_df['URL'].isin(data_links)]

# 将结果写入new_htmls.csv文件
new_htmls_df.to_csv('../Spider/data/new_htmls.csv', index=False, header=False)