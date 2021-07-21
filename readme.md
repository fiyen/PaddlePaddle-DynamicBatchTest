# 尝试动态BatchSize效果
不同于计算机视觉上的处理方式，文本样本一般为一个序列，而其对应的编码通常是不等长的。在构建一个批输入时，通常有两种处理方式：一种是直接将所有样本给填充成等长的编码序列，即编码后的输入长度都相等；另一种是在一个批的数据中将样本填充成等长的编码序列。显然，由于第一种方式将所有样本都填充到指定的长度，一些本来比较短的输入会和长序列的输入一样占用相同的显存空间，这种处理方式是不经济的。而另一种方式，当一个批中的所有样本都很短时，批长度也很短；当一个批中的某样本很长时，该批的长度会很长。由于批的样本数量通常是固定的，因此显存的占用可能出现较大的波动。有时在训练一个模型时，刚开始训练的好好的，中途突然出现显存溢出，可能就是这个原因。由于显存的波动，能开始训练不一定意味着能训练完。那么，自然会有人问，如果将批大小改为动态的，即批的样本数量是可变的，批长度长则样本数少，批长度短则样本数多，这样显存的利用率就不会出现波动了。那么这样处理的方式会造成其他问题吗？这里我们就进行一下探索。

**本人的研究方向是基于复杂网络的文本处理方法，也在致力于探究与深度学习相结合的方法。我会不定期更新自己的工作，如果有研究方向相同的，或者感兴趣的朋友，欢迎三连支持一下。[来AI Studio互粉吧~等你哦~](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/300157)**

### **更多项目链接点击[没入门的研究生的项目合集](https://aistudio.baidu.com/aistudio/projectdetail/542430)**

## 1. 定义动态BatchSize的数据读取器
为了实现这个操作，我们首先定义一下支持动态BatchSize的数据读取器。我们重新定义一个参数controller，来表示对batch_size和input_length的整合。显存的占用大致符合 类似batch_size * input_length * input_length的增长规律，因此我们就定义 
$controller=batch\_size*input\_length^2$

以下定义工具的参数解释如下：

data: 可迭代返回样本的数据，可以是list,Dataset,MapDataset等的实例；

controller: 如上所示的新引入的参数；

uprank: 是否按照数据长度进行升序排列，这里有三个参数，None表示不做升序或降序排列；True表示做升序排列；False表示做降序排列。默认为uprank=None；

ref_index: 进行升序或降序排列时所参照的样本数据所在的维度，默认为ref_index=0；

shuffle：是否对数据进行打乱操作。注意，如果需要对数据进行打乱，需要uprank=None且shuffle=True, 如果uprank不为None,则shuffle无效；


```python
!pip install --upgrade paddlenlp
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting paddlenlp
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/62/10/ccc761d3e3a994703f31a4d0f93db0d13789d1c624a0cbbe9fe6439ed601/paddlenlp-2.0.5-py3-none-any.whl (435kB)
         |████████████████████████████████| 440kB 13.6MB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.42.1)
    Requirement already satisfied, skipping upgrade: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.2.0)
    Requirement already satisfied, skipping upgrade: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.4.4)
    Requirement already satisfied, skipping upgrade: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (4.1.0)
    Requirement already satisfied, skipping upgrade: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (1.2.2)
    Requirement already satisfied, skipping upgrade: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.70.11.1)
    Requirement already satisfied, skipping upgrade: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.9.0)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.8.53)
    Requirement already satisfied, skipping upgrade: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (7.1.2)
    Requirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.22.0)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.15.0)
    Requirement already satisfied, skipping upgrade: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.5)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.0.0)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.8.2)
    Requirement already satisfied, skipping upgrade: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.2.3)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.14.0)
    Requirement already satisfied, skipping upgrade: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.20.3)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.21.0)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp) (0.24.2)
    Requirement already satisfied, skipping upgrade: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp) (0.3.3)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (0.18.0)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (3.9.9)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2019.9.11)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (1.25.6)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (3.0.4)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2.8)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied, skipping upgrade: pytz>=2017.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl->paddlenlp) (2019.3)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.10.1)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.6.0)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.23)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.2.0)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.6.1)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (2.4.2)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.0)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.4)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (16.7.9)
    Requirement already satisfied, skipping upgrade: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (5.1.2)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (2.0.1)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.4.10)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (7.0)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (0.16.0)
    Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (1.6.3)
    Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (2.1.0)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (0.14.1)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.5->Flask-Babel>=1.0.0->visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp) (0.6.0)
    Requirement already satisfied, skipping upgrade: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl->paddlenlp) (56.2.0)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp) (7.2.0)
    Installing collected packages: paddlenlp
      Found existing installation: paddlenlp 2.0.1
        Uninstalling paddlenlp-2.0.1:
          Successfully uninstalled paddlenlp-2.0.1
    Successfully installed paddlenlp-2.0.5



```python
from paddle.io import Dataset, DataLoader
from paddlenlp.data import Pad, Stack, Tuple
import numpy as np


class DatasetForDBL(Dataset):
  def __init__(self, data, controller, uprank=None, ref_index=0, shuffle=False):
    super().__init__()
    self._controller = controller
    self._uprank = uprank
    self._data = data
    self._ref_index = ref_index
    self._shuffle = shuffle
    self._init_dynamic_params()

    assert isinstance(data[0][ref_index], np.ndarray), 'The input data of ref_index is supposed to be np.ndarray.'

    self._rank()
    self._update_data_index()

  def __getitem__(self, idx):
    idx = self._index_to_index[idx]
    start, end = self._index[idx]
    return [self._data[i] for i in range(start, end)]

  def __len__(self):
    return len(self._index)

  def _init_dynamic_params(self):
    self._max_length = 0
    self._batch_size = 0

  def _update_data_index(self):
    self._index = []
    batch = 0
    start_idx = 0
    for idx, sample in enumerate(self._data):
      sample_len = sample[self._ref_index].size
      self._max_length = sample_len if sample_len > self._max_length else self._max_length
      self._batch_size = int(self._controller / self._max_length)
      batch += 1
      if batch > self._batch_size:
        self._index.append((start_idx, idx))
        start_idx = idx
        self._init_dynamic_params()
        self._max_length = sample_len
        self._batch_size = int(self._controller / self._max_length**2)
        batch = 0
    self._index.append((start_idx, len(self._data)))
    self._index_to_index = list(range(len(self._index)))
    if self._shuffle:
      random.shuffle(self._index_to_index)

  def _rank(self):
    if self._uprank is None:
      if self._shuffle:
        indexes = list(range(len(self._data)))
        random.shuffle(indexes)
        self._data = [self._data[i] for i in indexes]
      return
    if self._uprank:
      self._data = sorted(self._data, key=lambda x: x[self._ref_index].size)
    else:
      self._data = sorted(self._data, key=lambda x: x[self._ref_index].size, reverse=True)
```


```python
# 定义函数返回读取器
def get_dynamic_batch_loader(data, controller, collate_fn, uprank=None, ref_index=0, shuffle=False, places=None):
  dataset4dbl = DatasetForDBL(data, controller, uprank, ref_index, shuffle)
  return DataLoader(dataset4dbl, batch_size=None, collate_fn=collate_fn, places=places)
```

## 2. 随机样本测试
首先我们定义随机样本，来测试动态BatchSize在不同情境下的适应情况。这里，我们采用预训练的Bert模型，并随机生成一个16分类的数据样本，样本长度从2 - 256不等，样本字符数量与BertTokenizer的相同，样本数量为10000。


```python
from paddle.io import Dataset
import numpy as np


class RandomDataset(Dataset):
    def __init__(self, length_range=(2, 256), 
                       num_class=16, 
                       num_sample=10000, 
                       num_token=21128,
                       for_test=False):
        super().__init__()
        self._length_range = length_range
        self._num_class = num_class
        self._num_token = num_token
        self._for_test = for_test
        self._data = [self._gen_rand_sample() for _ in range(num_sample)]
        if not for_test:
            self._y = [np.random.randint(0, self._num_class) for _ in range(num_sample)]

    def __getitem__(self, idx):
        if self._for_test:
            return np.array(self._data[idx], dtype='int64')
        else:
            y = self._y[idx]
            y = np.array(y, dtype='int64')
            return np.array(self._data[idx], dtype='int64'), y
    
    def __len__(self):
        return len(self._data)

    def _gen_rand_sample(self):
        length = np.random.randint(self._length_range[0], self._length_range[1] + 1)
        sample = np.random.random_integers(1, self._num_token - 1, (length,))
        return sample

```


```python
# 纪录
import visualdl

writer = visualdl.LogWriter(logdir='log')
```

### 2.0. 测试情景零：正常情况
正常情况采用固定BatchSize的情况训练


```python
from paddle.io import DataLoader

rand_samps = RandomDataset()
collate_fn = lambda samples, fn=Tuple(Pad(pad_val=0, axis=0), Stack()): [data for data in fn(samples)]

fx_bs_loader = DataLoader(rand_samps, batch_size=20, collate_fn=collate_fn)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel_launcher.py:33: DeprecationWarning: This function is deprecated. Please call randint(1, 21127 + 1) instead



```python
# 训练
import paddlenlp
import paddle
import numpy as np

bert_model_0 = paddlenlp.transformers.BertForSequenceClassification.from_pretrained('bert-base-chinese', num_classes=16)

optimizer = paddle.optimizer.Adam(learning_rate=5e-3, parameters=bert_model_0.parameters())
loss_fn = paddle.nn.functional.cross_entropy

epochs = 2

train_step = 0
for epoch in range(epochs):
    for batch_id, (batch_x, batch_y) in enumerate(fx_bs_loader()):
        batch_x = paddle.to_tensor(batch_x)
        batch_y = paddle.to_tensor(batch_y)
        batch_y = paddle.unsqueeze(batch_y, axis=-1)
        out = bert_model_0(batch_x)
        # 计算损失
        loss = loss_fn(out, batch_y)
        # 损失后向传播
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        # 纪录
        train_step += 1
        writer.add_scalar('train_loss_0', loss, train_step)
        writer.add_scalar('batch_size_0', batch_x.shape[0], train_step)
        
        if batch_id % 100 == 0:
            print('Train epoch %d, batch %d, loss: %f' % (epoch+1, train_step, loss))
```

### 2.1. 测试情景一：打乱情况
由于样本是随机生成的，所以正常情况也即shuffle=True的情况，两者差距不大。


```python
from paddle.io import DataLoader

rand_samps = RandomDataset()
collate_fn = lambda samples, fn=Tuple(Pad(pad_val=0, axis=0), Stack()): [data for data in fn(samples)]
controller = 5000

dy_bs_loader_1 = get_dynamic_batch_loader(rand_samps, controller, collate_fn, uprank=None)
```


```python
# 训练
import paddlenlp
import paddle
import numpy as np

bert_model_1 = paddlenlp.transformers.BertForSequenceClassification.from_pretrained('bert-base-chinese', num_classes=16)

optimizer = paddle.optimizer.Adam(learning_rate=5e-3, parameters=bert_model_1.parameters())
loss_fn = paddle.nn.functional.cross_entropy

epochs = 2

train_step = 0
for epoch in range(epochs):
    for batch_id, (batch_x, batch_y) in enumerate(dy_bs_loader_1()):
        batch_x = paddle.to_tensor(batch_x)
        batch_y = paddle.to_tensor(batch_y)
        batch_y = paddle.unsqueeze(batch_y, axis=-1)
        out = bert_model_1(batch_x)
        # 计算损失
        loss = loss_fn(out, batch_y)
        # 损失后向传播
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        # 纪录
        train_step += 1
        writer.add_scalar('train_loss_1', loss, train_step)
        writer.add_scalar('batch_size_1', batch_x.shape[0], train_step)
        
        if batch_id % 100 == 0:
            print('Train epoch %d, batch %d, loss: %f' % (epoch+1, train_step, loss))
```

### 2.2. 测试情景二：按样本长度排序
这里只测试升序的情况。


```python
from paddle.io import DataLoader

rand_samps = RandomDataset()
collate_fn = lambda samples, fn=Tuple(Pad(pad_val=0, axis=0), Stack()): [data for data in fn(samples)]
controller = 5000

dy_bs_loader_2 = get_dynamic_batch_loader(rand_samps, controller, collate_fn, uprank=True)
```


```python
# 训练
import paddlenlp
import paddle
import numpy as np

bert_model_2 = paddlenlp.transformers.BertForSequenceClassification.from_pretrained('bert-base-chinese', num_classes=16)

optimizer = paddle.optimizer.Adam(learning_rate=5e-3, parameters=bert_model_2.parameters())
loss_fn = paddle.nn.functional.cross_entropy

epochs = 2

train_step = 0
for epoch in range(epochs):
    for batch_id, (batch_x, batch_y) in enumerate(dy_bs_loader_2()):
        batch_x = paddle.to_tensor(batch_x)
        batch_y = paddle.to_tensor(batch_y)
        batch_y = paddle.unsqueeze(batch_y, axis=-1)
        out = bert_model_2(batch_x)
        # 计算损失
        loss = loss_fn(out, batch_y)
        # 损失后向传播
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        # 纪录
        train_step += 1
        writer.add_scalar('train_loss_2', loss, train_step)
        writer.add_scalar('batch_size_2', batch_x.shape[0], train_step)
        
        if batch_id % 100 == 0:
            print('Train epoch %d, batch %d, loss: %f' % (epoch+1, train_step, loss))
```

结果如下：

<img src="https://ai-studio-static-online.cdn.bcebos.com/8125f21829914b9688c9e7734ad596c9dec9ba56cb804cb9977ec2b624e17640" width=400 height=400>
<img src="https://ai-studio-static-online.cdn.bcebos.com/69b1258a88874f81bf0fc8cccee0251b4a676d9fa6db42f18e5ba42e0f84ad03" width=400 height=400>

<img src="https://ai-studio-static-online.cdn.bcebos.com/c14756c08ad5456a93af11dea76513d5fc7e772723334d6187009cc147c7c156" width=400 height=400>
<img src="https://ai-studio-static-online.cdn.bcebos.com/2d3086fd88ad437ba4462f823dd105bc9ca85ff1d8904480afa88a5a974f86c9" width=400 height=400>

<img src="https://ai-studio-static-online.cdn.bcebos.com/d54de37280dc4d5ea0cf22725747d1625d4e55bd5d664036a58fe442af0b6b89" width=400 height=400>
<img src="https://ai-studio-static-online.cdn.bcebos.com/6f3e09e7608042cb8750a9e3da50a5019a33f7ae97084517bfa98f68e47cddc9" width=400 height=400>


```python

```

观察以上结果图，可以大致看出，动态BatchSize在显存利用率上更高，训练步数更短，速度更快。

## 3. 真实数据测试
在这里，我们用情感分析相关数据完成序列分类（SequenceClassification）来测试实际数据中不同情景下的训练效果。


```python
# 定义序列分类和词分类的数据集
from paddle.io import Dataset
from paddlenlp.data import Tuple, Pad, Stack
import paddlenlp
import random
import numpy as np


class RealDataset(Dataset):
    def __init__(self, data, label, tokenizer, max_seq_len=512, for_test=False):
        super().__init__()
        self._data = data
        self._label = label
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._for_test = for_test
    
    def __getitem__(self, idx):
        x = self._tokenizer.encode(self._data[idx], max_seq_len=self._max_seq_len)['input_ids']
        if self._for_test:
            return np.array(x, dtype='int64')
        else:
            y = self._label[idx]
            y = np.array(y, dtype='int64')
            return np.array(x, dtype='int64'), y
    
    def __len__(self):
        return len(self._data)


real_dataset = paddlenlp.datasets.load_dataset('chnsenticorp', splits=('train'), lazy=False)

real_data = [d['text'] for d in real_dataset]
real_label = [d['label'] for d in real_dataset]

# 打乱
indexes = list(range(len(real_data)))
random.shuffle(indexes)
real_data = [real_data[i] for i in indexes]
real_label = [real_label[i] for i in indexes]

tokenizer = paddlenlp.transformers.BertTokenizer.from_pretrained('bert-base-chinese')
```

    [2021-07-11 23:21:07,819] [    INFO] - Found /home/aistudio/.paddlenlp/models/bert-base-chinese/bert-base-chinese-vocab.txt



```python
class DatasetForDBL(Dataset):
  def __init__(self, data, controller, uprank=None, ref_index=0, shuffle=False):
    super().__init__()
    self._controller = controller
    self._uprank = uprank
    self._data = data
    self._ref_index = ref_index
    self._shuffle = shuffle
    self._init_dynamic_params()

    assert isinstance(data[0][ref_index], np.ndarray), 'The input data of ref_index is supposed to be np.ndarray.'

    self._rank()
    self._update_data_index()

  def __getitem__(self, idx):
    idx = self._index_to_index[idx]
    start, end = self._index[idx]
    return [self._data[i] for i in range(start, end)]

  def __len__(self):
    return len(self._index)

  def _init_dynamic_params(self):
    self._max_length = 0
    self._batch_size = 0

  def _update_data_index(self):
    self._index = []
    batch = 0
    start_idx = 0
    for idx, sample in enumerate(self._data):
      sample_len = sample[self._ref_index].size
      self._max_length = sample_len if sample_len > self._max_length else self._max_length
      self._batch_size = int(self._controller / self._max_length)
      batch += 1
      if batch > self._batch_size:
        self._index.append((start_idx, idx))
        start_idx = idx
        self._init_dynamic_params()
        self._max_length = sample_len
        self._batch_size = int(self._controller / self._max_length**2)
        batch = 0
    self._index.append((start_idx, len(self._data)))
    self._index_to_index = list(range(len(self._index)))
    if self._shuffle:
      random.shuffle(self._index_to_index)

  def _rank(self):
    if self._uprank is None:
      if self._shuffle:
        indexes = list(range(len(self._data)))
        random.shuffle(indexes)
        self._data = [self._data[i] for i in indexes]
      return
    if self._uprank:
      self._data = sorted(self._data, key=lambda x: x[self._ref_index].size)
    else:
      self._data = sorted(self._data, key=lambda x: x[self._ref_index].size, reverse=True)


# 定义函数返回读取器
def get_dynamic_batch_loader(data, controller, collate_fn, uprank=None, ref_index=0, shuffle=False, places=None):
  dataset4dbl = DatasetForDBL(data, controller, uprank, ref_index, shuffle)
  return DataLoader(dataset4dbl, batch_size=None, collate_fn=collate_fn, places=places)
```


```python
# 纪录
import visualdl

writer = visualdl.LogWriter(logdir='log4real')
```

### 3.0. 正常情景
正常情景即使用固定BatchSize的数据读取器。


```python
from paddle.io import DataLoader

real_dataset = RealDataset(real_data, real_label, tokenizer, 512)
collate_fn = lambda samples, fn=Tuple(Pad(pad_val=0, axis=0), Stack()): [data for data in fn(samples)]

fx_bs_loader = DataLoader(real_dataset, batch_size=20, collate_fn=collate_fn)
```


```python
# 训练
import paddlenlp
import paddle
import numpy as np

bert_model_0 = paddlenlp.transformers.BertForSequenceClassification.from_pretrained('bert-base-chinese', num_classes=2)

optimizer = paddle.optimizer.Adam(learning_rate=5e-6, parameters=bert_model_0.parameters())
loss_fn = paddle.nn.functional.cross_entropy
acc_fn = paddle.metric.accuracy

epochs = 2

train_step = 0
for epoch in range(epochs):
    acc = 0
    steps = 0
    for batch_id, (batch_x, batch_y) in enumerate(fx_bs_loader()):
        batch_x = paddle.to_tensor(batch_x)
        batch_y = paddle.to_tensor(batch_y)
        batch_y = paddle.unsqueeze(batch_y, axis=-1)
        out = bert_model_0(batch_x)
        # 计算损失
        loss = loss_fn(out, batch_y)
        # 计算准确率
        tem_acc = acc_fn(out, batch_y)
        acc = (acc * steps + tem_acc * batch_x.shape[0]) / (steps + batch_x.shape[0])
        steps += batch_x.shape[0]
        # 损失后向传播
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        # 纪录
        train_step += 1
        writer.add_scalar('real_train_loss_0', loss, train_step)
        writer.add_scalar('real_batch_size_0', batch_x.shape[0], train_step)
        writer.add_scalar('real_accuracy_0', acc, train_step)
        
        if batch_id % 100 == 0:
            print('Train epoch %d, batch %d, loss: %.6f, acc: %.6f' % (epoch+1, train_step, loss, acc))
```

    [2021-07-11 23:21:17,134] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/bert-base-chinese/bert-base-chinese.pdparams
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))


    Train epoch 1, batch 1, loss: 0.715938, acc: 0.550000
    Train epoch 1, batch 101, loss: 0.260708, acc: 0.832673
    Train epoch 1, batch 201, loss: 0.143788, acc: 0.870896
    Train epoch 1, batch 301, loss: 0.302375, acc: 0.883223
    Train epoch 1, batch 401, loss: 0.085725, acc: 0.890898
    Train epoch 2, batch 481, loss: 0.277026, acc: 0.900000
    Train epoch 2, batch 581, loss: 0.249776, acc: 0.930198
    Train epoch 2, batch 681, loss: 0.104511, acc: 0.938806
    Train epoch 2, batch 781, loss: 0.125206, acc: 0.937376
    Train epoch 2, batch 881, loss: 0.036320, acc: 0.939776


### 3.1. 打乱情景
使用动态BatchSize，数据打乱。


```python
from paddle.io import DataLoader

real_dataset = RealDataset(real_data, real_label, tokenizer, 1024)
collate_fn = lambda samples, fn=Tuple(Pad(pad_val=0, axis=0), Stack()): [data for data in fn(samples)]
controller = 5000

dy_bs_loader_1 = get_dynamic_batch_loader(real_dataset, controller, collate_fn, uprank=None, shuffle=True)
```


```python
# 训练
import paddlenlp
import paddle
import numpy as np

bert_model_1 = paddlenlp.transformers.BertForSequenceClassification.from_pretrained('bert-base-chinese', num_classes=2)
clip = paddle.nn.ClipGradByValue(400)
optimizer = paddle.optimizer.Adam(learning_rate=5e-6, parameters=bert_model_1.parameters(), grad_clip=clip)
loss_fn = paddle.nn.functional.cross_entropy
acc_fn = paddle.metric.accuracy

epochs = 2

train_step = 0
for epoch in range(epochs):
    acc = 0
    steps = 0
    for batch_id, (batch_x, batch_y) in enumerate(dy_bs_loader_1()):
        batch_x = paddle.to_tensor(batch_x)
        batch_y = paddle.to_tensor(batch_y)
        batch_y = paddle.unsqueeze(batch_y, axis=-1)
        out = bert_model_1(batch_x)
        # 计算损失
        loss = loss_fn(out, batch_y)
        # 计算准确率
        tem_acc = acc_fn(out, batch_y)
        acc = (acc * steps + tem_acc * batch_x.shape[0]) / (steps + batch_x.shape[0])
        steps += batch_x.shape[0]
        # 损失后向传播
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        # 纪录
        train_step += 1
        writer.add_scalar('real_train_loss_1', loss, train_step)
        writer.add_scalar('real_batch_size_1', batch_x.shape[0], train_step)
        writer.add_scalar('real_accuracy_1', acc, train_step)
        
        if batch_id % 100 == 0:
            print('Train epoch %d, batch %d, loss: %.6f, acc: %.6f' % (epoch+1, train_step, loss, acc))
```

    [2021-07-11 23:07:16,298] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/bert-base-chinese/bert-base-chinese.pdparams
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))


    Train epoch 1, batch 1, loss: 0.754673, acc: 0.368421
    Train epoch 1, batch 101, loss: 0.195543, acc: 0.781171
    Train epoch 1, batch 201, loss: 0.283495, acc: 0.837791
    Train epoch 1, batch 301, loss: 0.311484, acc: 0.863916
    Train epoch 1, batch 401, loss: 0.188655, acc: 0.877811
    Train epoch 1, batch 501, loss: 0.065736, acc: 0.886324
    Train epoch 2, batch 590, loss: 0.236028, acc: 0.947368
    Train epoch 2, batch 690, loss: 0.044536, acc: 0.926605
    Train epoch 2, batch 790, loss: 0.225879, acc: 0.931559
    Train epoch 2, batch 890, loss: 0.318091, acc: 0.937891
    Train epoch 2, batch 990, loss: 0.115591, acc: 0.941912
    Train epoch 2, batch 1090, loss: 0.016826, acc: 0.943281


### 3.2. 按样本长度排序
只测试升序的情况。


```python
from paddle.io import DataLoader

real_dataset = RealDataset(real_data, real_label, tokenizer, 1024)
collate_fn = lambda samples, fn=Tuple(Pad(pad_val=0, axis=0), Stack()): [data for data in fn(samples)]
controller = 5000

dy_bs_loader_2 = get_dynamic_batch_loader(real_dataset, controller, collate_fn, uprank=True)
```


```python
# 训练
import paddlenlp
import paddle
import numpy as np

bert_model_2 = paddlenlp.transformers.BertForSequenceClassification.from_pretrained('bert-base-chinese', num_classes=2)

clip = paddle.nn.ClipGradByValue(400)
optimizer = paddle.optimizer.Adam(learning_rate=5e-6, parameters=bert_model_2.parameters(), grad_clip=clip)
loss_fn = paddle.nn.functional.cross_entropy
acc_fn = paddle.metric.accuracy

epochs = 2

train_step = 0
for epoch in range(epochs):
    acc = 0
    steps = 0
    for batch_id, (batch_x, batch_y) in enumerate(dy_bs_loader_2()):
        batch_x = paddle.to_tensor(batch_x)
        batch_y = paddle.to_tensor(batch_y)
        batch_y = paddle.unsqueeze(batch_y, axis=-1)
        out = bert_model_2(batch_x)
        # 计算损失
        loss = loss_fn(out, batch_y)
        # 计算准确率
        tem_acc = acc_fn(out, batch_y)
        acc = (acc * steps + tem_acc * batch_x.shape[0]) / (steps + batch_x.shape[0])
        steps += batch_x.shape[0]
        # 损失后向传播
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        # 纪录
        train_step += 1
        writer.add_scalar('real_train_loss_2', loss, train_step)
        writer.add_scalar('real_batch_size_2', batch_x.shape[0], train_step)
        writer.add_scalar('real_accuracy_2', acc, train_step)
        
        if batch_id % 100 == 0:
            print('Train epoch %d, batch %d, loss: %.6f, acc: %.6f' % (epoch+1, train_step, loss, acc))
```

    [2021-07-11 23:02:16,449] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/bert-base-chinese/bert-base-chinese.pdparams
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))


    Train epoch 1, batch 1, loss: 0.629536, acc: 0.703125
    Train epoch 1, batch 101, loss: 0.390604, acc: 0.833605
    Train epoch 1, batch 201, loss: 0.529718, acc: 0.856681
    Train epoch 2, batch 205, loss: 0.378667, acc: 0.854167
    Train epoch 2, batch 305, loss: 0.377293, acc: 0.916191
    Train epoch 2, batch 405, loss: 0.238848, acc: 0.925107


结果如下：

<img src="https://ai-studio-static-online.cdn.bcebos.com/e9fc7f1f0b6148b9abdb5c72b97333689d63308235824887b3803d02657ad2df" width=300 height=300>
<img src="https://ai-studio-static-online.cdn.bcebos.com/b182f68c56af4b398a967a05c9f5b7e77cbd9f19420e4252b3f0730f45c58122" width=300 height=300>
<img src="https://ai-studio-static-online.cdn.bcebos.com/e788f586c3284b7ca49e9a62d34fe3a1169d0ba547b34ca0aeef11fa081d7a14" width=300 height=300>

<img src="https://ai-studio-static-online.cdn.bcebos.com/081034f7076e42f4b5b6a82025845322f90982b406cb49c5ba88042cfb0710cf" width=300 height=300>
<img src="https://ai-studio-static-online.cdn.bcebos.com/043358a574dd403f8d119bc0ea50c78496f26cdc76064c2a9cc9e927efcf0a2f" width=300 height=300>
<img src="https://ai-studio-static-online.cdn.bcebos.com/f83ed517df1e4f1bbf4666a2d08a91549d902e2fb11a4f8a88ed0b6e74e8178d" width=300 height=300>

<img src="https://ai-studio-static-online.cdn.bcebos.com/828aba60ee96489681d5bb52fa856d357a4cd8d9fe1e43dc847e3d60e3555e69" width=300 height=300>
<img src="https://ai-studio-static-online.cdn.bcebos.com/b3db6fa1872f4c2cb95e979aefd1dae9d4d5cdd954244b68a019f919f028e29e" width=300 height=300>
<img src="https://ai-studio-static-online.cdn.bcebos.com/13e3648591a048dba815e437459d58f208b49254fa024f3ca282e1a411205ed2" width=300 height=300>


## 总结
从随机样本和真实样本中，我们都可以看到，动态BatchSize的训练步长要小于固定BatchSize,且训练总时长较短（真实样本中随机排序的步长反而增加，是因为样本最长长度增加到了1024，而固定的长度是512）。训练的损失波动不大，同时训练的准确率也没有太大的影响。因此可以看出，适当使用动态BatchSize对训练速度是有提升的。但是这里需要提出结果以外的问题：动态BatchSize会出现训练不稳定的情况，即某些情况下训练过程中会出现参数为NaN的情况，这些通过调小学习率和增加grad_clip会有一定程度的缓解，但是并不能从根本上解决问题。打乱数据并重复训练可能会不再出现此问题，但具体原因有待进一步探讨。此外，对于多卡训练的效果也有待验证，感兴趣的可以关注后续进展。
