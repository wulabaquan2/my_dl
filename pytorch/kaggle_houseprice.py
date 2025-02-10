import numpy as np
import pandas as pd
import torch
from torch import nn
import d2l 
import hashlib
import os
import tarfile
import zipfile
import requests

'''
数据预处理-不同尺度、数据缺失(nan)、类别数据的预处理
数据上下限差异大时考量相对误差的损失评价-取对数后计算均方根误差-log_rmse  #不是训练用的损失函数,是质量评估的损失计算
'''
#下载kaggle数据
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    loss=nn.MSELoss()
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

#训练过程
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    loss=nn.MSELoss()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)  #训练用mse
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))    #质量评估用log_rmse
        #if epoch%5==0:
        #    print(f'epoch:{epoch}, loss:{float(train_ls[-1]):f}')
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

#删除无用数据第1列是ID,仅代表顺序
#test_data中不包含SalePrice
#train_data删除最后一列(SalePrice)
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index #获取数值数据
#*不同尺度数据-每种特征的数值数据标准化为 (x-u)/σ
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
#*nan数据-设置为0，依赖前面的标准化
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
#*类别数据- 类别数据转为独热编码(特征维数扩充)
#问题：当类别过多时，稀疏矩阵占用大量内存
'''
示例: kaggle_house_train数据中MSZoning是枚举类别,转成one-hot后数据维度增加枚举项的个数(1->6)
    MSZoning       ->   MSZonging_C MSZonging_FV MSZonging_RL MSZonging_RH MSZonging_RM MSZonging_nan
0    C                      1           0           0           0           0               0
1    FV                     0           1           0           0           0               0
2    RL                     0           0           1           0           0               0
3    RH                     0           0           0           1           0               0
4    RM                     0           0           0           0           1               0
#列(nan)在dummy_na=true时为单独一列
'''
#将类别数据转为独热编码
all_features = pd.get_dummies(all_features, dummy_na=True,dtype=float)
'''
all_features.shape[1]由79扩充为330,其中43个类别数据扩展独热编码提供了(330-79)个特征(包含nan的类别提供classNum个,不包含nan的类别提供classNum+1个)
'''
#pandas类型转为tensor类型
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64  #默认超参
train_l_sum, valid_l_sum = 0, 0
#k折交叉训练: 训练集->1/k的验证集 (k-1)/k的训练集
for i in range(k):
    data = d2l.get_k_fold_data(k, i, train_features, train_labels)
    net = nn.Sequential(nn.Linear(train_features.shape[1],1))
    #返回每个epoch的损失
    train_ls, valid_ls = train(net, *data, num_epochs, lr,
                                weight_decay, batch_size)
    train_l_sum += train_ls[-1]
    valid_l_sum += valid_ls[-1]

    print(f'折{i + 1}，训练log rmse {float(train_ls[-1]):f}, '
            f'验证log rmse {float(valid_ls[-1]):f}')

#k折交叉训练的平均损失
print(f'{k}-折验证: 平均训练log rmse: {float(train_l_sum / k):f}, '
      f'平均验证log rmse: {float(valid_l_sum / k):f}')

'''
默认超参
折1，训练log rmse0.169899, 验证log rmse0.156711
折2，训练log rmse0.162596, 验证log rmse0.191471
折3，训练log rmse0.163348, 验证log rmse0.168230
折4，训练log rmse0.167998, 验证log rmse0.154813
折5，训练log rmse0.162411, 验证log rmse0.182453
5-折验证: 平均训练log rmse: 0.165250, 平均验证log rmse: 0.170736
调整超参

'''

