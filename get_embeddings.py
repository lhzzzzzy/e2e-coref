#pip install gensim  这个库记得装
from gensim.models import KeyedVectors 
import numpy as np
import jieba


file = 'embedding.txt' 
w2v_model = KeyedVectors.load_word2vec_format(file, binary=False)
# 载入模型
w2v_model.save('Tencent_AILab_ChineseEmbedding.bin')
w2v_model = KeyedVectors.load('Tencent_AILab_ChineseEmbedding.bin')
# 加载模型

# 获取词的向量表示 100维向量
# 组合词向量

# words换成句子的字词就行
words = ['及时雨', '宋江']
vector = {}
for word in words:
    if word not in w2v_model:
        wor = jieba.lcut(place)
        v = np.zeros((100,), dtype='float32')
        for w in wor:
            v += w2v_model[word]
        vector[word] = v / len(wor)
    else:
        vector[word] = w2v_model[word]