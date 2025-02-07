import torch
import d2l
from torch import nn

'''
分类(全连接层->2个隐藏层+dropout)-过拟合优化-隐藏层输出增加dropout
*自定义net类,其中用源码实现模型训练时前向传播中每个隐藏层输出添加dropout
*用现成net类
'''

#定义dropout层
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)  #在标准暂退法正则化中，通过按保留（未丢弃）的节点的分数进行规范化来消除每一层的偏差: 丢弃节点会导致值变小,通过1/(1-p)恢复期望值

dropout1, dropout2 = 0.2, 0.5

#源码实现模型训练时前向传播中每个隐藏层输出添加dropout
#问题: Linear的weight默认初始化是什么
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
#自定义net类,
#net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
#用现成net类
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

'''
epoch: 0, loss:0.8694607019424438, acc:0.6735666666666666
epoch: 1, loss:0.5290234684944153, acc:0.80565
epoch: 2, loss:0.45933470129966736, acc:0.8311333333333333
epoch: 3, loss:0.4210740327835083, acc:0.8455666666666667
epoch: 4, loss:0.39578455686569214, acc:0.8549833333333333
epoch: 5, loss:0.3767194449901581, acc:0.86125
epoch: 6, loss:0.36386004090309143, acc:0.8665666666666667
epoch: 7, loss:0.3497835099697113, acc:0.87165
epoch: 8, loss:0.3411303758621216, acc:0.8741333333333333
epoch: 9, loss:0.33042874932289124, acc:0.8788
'''