# Task 3 基于机器学习的文本分类

2群-181-datago-StarLEE

## 1. Word Embedding

### 1.1 One-hot

将每个单词使用一个离散向量表示。若有*x*个不同的字符就用*x*个*x*维的单位向量来表示。

### 1.2 Bag of Words

也称作Count Vectors，每个文档的字词用出现次数来表示

sklearn库中`CountVectorizer`可便捷实现词袋

***图片1***

### 1.3 N-gram

N-gram加入了相邻N个单词的组合，并进行计数

### 1.4 TF-IDF

TF-IDF 分为两部分：**词语频率**（Term Frequency）和**逆文档频率**（Inverse Document Frequency）。

TF(t) = 该词语在当前文档频数 / 当前文档词语总数
IDF(t) = ln（文档总数 / 出现该词语文档总数）

## 2. 基于ML的文本分类

### 2.1 Count Vectors + Ridge Classifier

```python
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('../data/train_set.csv', sep='\t')

vectorizer = CountVectorizer(max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.748813
```

### 2.2 TF-IDF + Ridge Classifier

```python
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('../input/train_set.csv', sep='\t', nrows=15000)

tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.87
```

## 作业

1.尝试改变TF-IDF参数，并验证精度









2.尝试使用其他机器学习模型，完成训练和验证