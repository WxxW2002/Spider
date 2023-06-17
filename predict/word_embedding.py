import pandas as pd
import jieba
from gensim.models import Word2Vec

# 加载停用词
stopwords = set()
with open('data/stopwords/cn_stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f:
        stopwords.add(line.strip())

data = pd.read_csv('data/house_processed.csv')

titles = data['Title'].tolist()
subtitles = data['Subtitle'].tolist()

tokenized_titles = []
for title, subtitle in zip(titles, subtitles):
    # 分词并去除停用词
    title_tokens = [token for token in jieba.cut(title) if token not in stopwords and token != ' ']
    subtitle_tokens = [token for token in jieba.cut(subtitle) if token not in stopwords and token != ' ']
    tokens = title_tokens + subtitle_tokens
    tokenized_titles.append(tokens)

model = Word2Vec(tokenized_titles, vector_size=100, window=5, min_count=1, workers=4)

word_vectors = model.wv

embedding_features = []
for tokens in tokenized_titles:
    embeddings = [word_vectors[token] for token in tokens if token in word_vectors]
    if embeddings:
        avg_embedding = sum(embeddings) / len(embeddings)
        embedding_features.append(avg_embedding)
    else:
        embedding_features.append([0.0] * 100)

embedding_features = pd.DataFrame(embedding_features, columns=[f'embedding_{i}' for i in range(100)])
data_with_embeddings = pd.concat([data, embedding_features], axis=1)

data_with_embeddings.to_csv('data/house_with_embeddings.csv', index=False)
