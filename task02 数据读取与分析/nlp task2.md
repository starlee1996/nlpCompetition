# Task 2 数据读取与数据分析

2群-181-datago-StarLEE

## 1. 数据读取

利用Pandas库中`read_csv()`可以对csv数据进行方便的读取和操作

```python
import pandas as pd
train_df = pd.read_csv('../train_set.csv', sep='\t')
train_df.head(10)
```

数据内容如下：

<img src="C:\Users\Yixin\AppData\Roaming\Typora\typora-user-images\image-20200722080757452.png" alt="image-20200722080757452" style="zoom: 80%;" />

label为文本对应类别，text为文本内容

## 2. 数据分析

主要目标有三：

+ 数据的新闻文本长度
+ 数据的类别分布状况
+ 数据的字符分布状况

### 2.1 文本长度分析

每行文本的字符通过空格相隔，通过对字符串进行`split(' ')`操作再对list进行计数便可以得到文本的长度

```python
%pylab inline
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())
```

输出结果如下：

<img src="C:\Users\Yixin\AppData\Roaming\Typora\typora-user-images\image-20200722082136267.png" alt="image-20200722082136267" style="zoom:80%;" />

相关的统计信息都很清晰。此外，还可以进一步进行可视化，构建文本长度的直方图：

```python
import matplotlib.pyplot as plt
_ = plt.hist(train_df['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("Histogram of char count")
```

<img src="C:\Users\Yixin\AppData\Roaming\Typora\typora-user-images\image-20200722082539953.png" alt="image-20200722082539953" style="zoom:80%;" />

### 2.2 文本类别分布

对数据label进行计数：

```python
train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")
```

<img src="C:\Users\Yixin\AppData\Roaming\Typora\typora-user-images\image-20200722083016808.png" alt="image-20200722083016808" style="zoom:80%;" />

可见，文本的种类分布是不均匀的。

### 2.3 字符分布分析

首先将train set中所有句子整合起来，然后再对字符进行计数。通过collections库中的Counter对元素进行计数并构建word_count字典，为了更方便分析，可以对字典进行排序：

```python
from collections import Counter
all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(' '))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)

print('Number of words:', len(word_count))
print('Word with highest frequency: ', word_count[0])
print('Word with lowest frequency: ', word_count[-1])
```

<img src="C:\Users\Yixin\AppData\Roaming\Typora\typora-user-images\image-20200722084242490.png" alt="image-20200722084242490" style="zoom:80%;" />

共有6869种字符，字符‘3750’词频最高，字符‘3133’词频最低。

进一步还可以通过字符的词频反推出标点符号

```python
train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(train_df['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse = True)

print(word_count[:6])
```

输出结果如下：

<img src="C:\Users\Yixin\AppData\Roaming\Typora\typora-user-images\image-20200722085855869.png" alt="image-20200722085855869" style="zoom:80%;" />

可见字符‘3750’，‘900’，‘648’占比约为99%，为标点符号的可能性很大。

# 课后作业

借助正则表达式re库，可以对文本进行基于多分隔符的分割，然后再计数就可以解决：

```python
import re
train_df['sentence_count'] = train_df['text'].apply(lambda x: len(re.split(r"3750|900|648", x)))

train_df['sentence_count'].describe()
```

结果如下：

<img src="C:\Users\Yixin\AppData\Roaming\Typora\typora-user-images\image-20200722093221340.png" alt="image-20200722093221340" style="zoom:80%;" />

进行可视化：

```python
_ = plt.hist(train_df['sentence_count'], bins=50)
plt.xlabel('Text sentence count')
plt.title("Histogram of sentence count")
```

<img src="C:\Users\Yixin\AppData\Roaming\Typora\typora-user-images\image-20200722093311540.png" alt="image-20200722093311540" style="zoom:80%;" />

可见文本的平均句子数量为81句