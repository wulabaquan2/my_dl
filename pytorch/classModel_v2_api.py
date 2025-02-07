import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
"""
分类(全连接层): 28*28图像计算10个类别的one-hot标签
模型架构: y=softmax(wx+b),类别=10,图像大小28*28(784)
    x: 1x784    #图像展开为1维向量
    w: 784*10   #每个类别都有对应的784个权重,共有10个类别
    b: 1*10     #每个类别都有1个偏置
    softmax: 将y(1*10)数据规范为为非负数并且总和为1，同时让模型保持 可导的性质
"""

#读取Fashion-MNIST数据集
def load_data_fashion_mnist(batch_size, resize=None): 
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在0～1之间
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))

#工具函数: 初始化Linear层的参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01) #mean默认是0

#工具函数: 小批次预测的y_hat中预测正确(与标签y相同)的数量
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

#定义一个epoch训练过程
def train_epoch_ch3(net, train_iter, loss, updater):  
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    loss_sum=0
    train_num=0
    train_accuracy=0
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])  
        loss_sum+=l.sum() #小批次损失累加至损失总和
        train_num+=y.numel()#小批次样本数累加至总训练样本数
        train_accuracy+=accuracy(y_hat,y)#小批次准确预测的数量累加至总准确预测数量
    # 返回平均训练损失和训练精度
    return loss_sum /train_num,train_accuracy/ train_num

batch_size=256
num_epochs=10
lr=0.1
num_inputs = 784
num_outputs = 10
train_iter, test_iter = load_data_fashion_mnist(batch_size)

#定义模型架构,其中softmax计算在loss函数中定义
net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))
#初始化模型参数
net.apply(init_weights) #apply: 遍历对象的children应用回调函数
#定义损失函数-交叉熵损失(自带softmax不论reduction是什么)
loss = nn.CrossEntropyLoss(reduction='none')
#定义优化函数-sgd
updater = torch.optim.SGD(net.parameters(), lr=0.1)
#训练全部epoch并打印每个epoch的平均损失和准确率
for epoch in range(num_epochs):
    train_loss,train_acc = train_epoch_ch3(net, train_iter, loss, updater)
    print(f'epoch: {epoch}, loss:{train_loss}, acc:{train_acc}')

#测试集结果对比
'''
epoch: 0, loss:0.7854874134063721, acc:0.7480833333333333
epoch: 1, loss:0.5694980025291443, acc:0.8140166666666667
epoch: 2, loss:0.5263306498527527, acc:0.8244333333333334
epoch: 3, loss:0.5015104413032532, acc:0.8321333333333333
epoch: 4, loss:0.48515257239341736, acc:0.837
epoch: 5, loss:0.47351279854774475, acc:0.8403666666666667
epoch: 6, loss:0.4648808240890503, acc:0.8424
epoch: 7, loss:0.4580745995044708, acc:0.8442833333333334
epoch: 8, loss:0.45166853070259094, acc:0.8474
epoch: 9, loss:0.4472154974937439, acc:0.8484666666666667
true:  tensor([9, 2, 1, 1, 6, 1])
pred:  tensor([9, 2, 1, 1, 6, 1])
'''
for X, y in test_iter:
    break
X=X[0:6]
y=y[0:6]
print('true: ',y)
print('pred: ',net(X).argmax(axis=1))