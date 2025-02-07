import torch
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
def load_data_fashion_mnist(batch_size, resize=None):  #@save
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

#定义softmax层
'''
softmax函数能够将未规范化的预测变换为非负数并且总和为1，同时让模型保持 可导的性质
'''
def softmax(X):
    X_exp = torch.exp(X)    #非负数
    partition = X_exp.sum(1, keepdim=True) 
    return X_exp / partition  #总和为1

#定义模型: 架构为y=softmax(wx+b)
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)  #W,b是全局变量

#定义损失函数: 交叉熵损失
#示例: softmax的结果为[0.1,0.2,0.7]与真实标签[0,0,1]的损失. 因为真实标签y是one-hot,所以损失=-log(0.7)
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y]) 

#创建优化函数: SGD-> param-=lr*grad/batch_size
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_() #等同于zero_grad()用于清理grad防止对下一批次的grad计算造成影响

#工具类: 用于优化函数改为优化类,将函数形参替换为成员变量初始化(方便优化函数调参)
class Updater():  
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size):  #与tensorflow不同: pytoch中params的属性包含grad,而tensorflow因为grad和params独立需要额外形参grad
        sgd(self.params, self.lr, batch_size)

#工具函数: 小批次预测的y_hat中预测正确(与标签y相同)的数量
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

#定义一个epoch训练过程
def train_epoch_ch3(net, train_iter, loss, updater):  
    # 训练损失总和、训练准确度总和、样本数
    loss_sum=0
    train_num=0
    train_accuracy=0
    for X, y in train_iter: #train_iter每次返回一个批次的所有数据
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()  #不同批次的梯度计算互无影响
            l.mean().backward()  #计算梯度
            updater.step()       #优化器根据梯度更新参数(updater在初始化时有绑定参数(全局变量W,b))
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()   
            updater(X.shape[0])  #X.shape[0]是因为sgd优化器的分母是batchsize
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
#28*28*1图像输出10个类别,模型架构: y=wx+b
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) #784*10
b = torch.zeros(num_outputs, requires_grad=True)   #1*10
#读取fashion_mnist数据集
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=None) 
#初始化模型参数
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
#初始化优化器
updater=Updater([W,b],lr)
#训练全部epoch并打印每个epoch的平均损失和准确率
for epoch in range(num_epochs):
    train_loss,train_acc = train_epoch_ch3(net, train_iter, cross_entropy, updater)
    print(f'epoch: {epoch}, loss:{train_loss}, acc:{train_acc}')

#测试集结果对比
'''
true: tensor([9, 2, 1, 1, 6, 1])
pred: tensor([9, 2, 1, 1, 6, 1])
'''
for X, y in test_iter:
    break
X=X[0:6]
y=y[0:6]
print('true: ',y)
print('pred: ',net(X).argmax(axis=1))
