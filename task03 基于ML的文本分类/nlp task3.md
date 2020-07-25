# Task 3 基于机器学习的文本分类

2群-181-datago-StarLEE

## 1. Word Embedding

### 1.1 One-hot

将每个单词使用一个离散向量表示。若有*x*个不同的字符就用*x*个*x*维的单位向量来表示。

### 1.2 Bag of Words

也称作Count Vectors，每个文档的字词用出现次数来表示

sklearn库中`CountVectorizer`可便捷实现词袋

<img src="https://github.com/starlee1996/nlpCompetition/blob/master/task03%20%E5%9F%BA%E4%BA%8EML%E7%9A%84%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB/pictures/1.png?raw=true" style="zoom:80%;" />

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
# 0.8768
```

## 作业

1.尝试改变TF-IDF参数，并验证精度

尝试改变了TF-IDF中`ngram_range`的取值，将其增大至`(4, 5)`

```python
tfidf = TfidfVectorizer(ngram_range=(4,5), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.6312
```

可见，单纯的增大N-gram，结合更多的相邻词语并不是一个很好的方法，f1 score明显降低了

2.尝试使用其他机器学习模型，完成训练和验证

尝试SVM和Random Forest

```python
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])

clf = svm.SVC()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.8723

clf = RandomForestClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.7764
```

可见SVM和Ridge Classifier的表现较好，进一步可以在深度学习中提高精度。