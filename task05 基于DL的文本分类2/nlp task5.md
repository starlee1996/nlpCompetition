# Task 4 基于深度学习的文本分类

2群-181-datago-StarLEE

## 1. 原文本表示方法的缺陷

对于One-hot，Bag of Words，N-gram以及TF-IDF这类文本表示方法，转换得到的文本向量维度很高，需要大量的训练时间，且未考虑单词间联系，只是进行了单纯 的统计。

深度学习可以将文本映射到一个低维空间，例如：FastText，WordsVec和Bert

## 2. FastText

FastText借助深度学习，通过Embedding层将单词映射到稠密空间，然后把单词在Embedding空间中进行平均，进而完成分类。FastText为一个三层的NN。

借助keras可以实现FastText：

```python
from __future__ import unicode_literals

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dense

VOCAB_SIZE = 2000
EMBEDDING_DIM = 100
MAX_WORDS = 500
CLASS_NUM = 5

def build_fastText():
	moedl = Sequential()
	model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length = MAX_WORDS))
	model.add(GLOBALAVERAGEPOOLING1D())
	model.add(Dense(CLASS_NUM, activation = 'softmax'))
	model.compile(loss='categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])
	return model

if __name__ == '__main__':
	model = build_fastText()
	print(model.summary())
```

FastText相较于TF-IDF，具有以下优点：

+ FastText的单词的Embedding叠加获得文档向量，将相似的句子分为一类
+ FastText学习到的Embedding空间维度较低，训练时间更短

## 3. 基于FastText的文本分类

### 3.1 分类模型

```python
import pandas as pd
from sklearn.metrics import f1_score

# 转换为FastText需要的格式
train_df = pd.read_csv('../input/train_set.csv', sep='\t', nrows=15000)
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text','label_ft']].iloc[:-5000].to_csv('train.csv', index=None, header=None, sep='\t')

import fasttext
model = fasttext.train_supervised('train.csv', lr=1.0, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=25, loss="hs")

val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))
```

### 3.2 验证集调参

使用10-fold cross-validation，基于验证集的结果调整超参数以优化模型

```python
label2id = {}
for i in range(total):
    label = str(all_labels[i])
    if label not in label2id:
        label2id[label] = [i]
    else:
        label2id[label].append(i)
```