import torch
import d2l
from torch import nn
'''
线性回归-过拟合优化-损失函数增加L2范数权重惩罚
'''

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

#L2范数权重惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2 #除2是抵消w^2的导数

#权重惩罚相关代码用源码实现
def train_src(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight},
        {"params":net[0].bias}], lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + wd * l2_penalty(net[0].weight)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            train_loss= d2l.evaluate_loss(net, train_iter, loss)
            test_loss=d2l.evaluate_loss(net, test_iter, loss)
            print(f'epoch: {epoch}, train_loss:{train_loss}, test_loss:{test_loss}')
    print('w的L2范数：', net[0].weight.norm().item())  #item(): 将此张量的值作为标准 Python 数字返回.这仅适用于包含一个元素的张量


#api实现
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            train_loss= d2l.evaluate_loss(net, train_iter, loss)
            test_loss=d2l.evaluate_loss(net, test_iter, loss)
            print(f'epoch: {epoch}, train_loss:{train_loss}, test_loss:{test_loss}')
    print('w的L2范数：', net[0].weight.norm().item())  #item(): 将此张量的值作为标准 Python 数字返回.这仅适用于包含一个元素的张量

#权重惩罚用源码实现

#不加权重惩罚
print('------不加权重惩罚------')
train_src(0)
#train_concise(0)

#权重惩罚=3
print('------权重惩罚=3------')
train_src(3)
#train_concise(3)


'''
结果
------不加权重惩罚------
epoch: 4, train_loss:12.3496244430542, test_loss:185.6776513671875
epoch: 9, train_loss:1.649522614479065, test_loss:186.55025970458985
epoch: 14, train_loss:0.2857592225074768, test_loss:187.13675857543944
epoch: 19, train_loss:0.05541698709130287, test_loss:187.41381912231446
epoch: 24, train_loss:0.01157348807901144, test_loss:187.53939483642577
epoch: 29, train_loss:0.0025443078484386206, test_loss:187.59837890625
epoch: 34, train_loss:0.0005813664000015705, test_loss:187.62638580322266
epoch: 39, train_loss:0.00013670250336872414, test_loss:187.6386877441406
epoch: 44, train_loss:3.282398283772636e-05, test_loss:187.643720703125
epoch: 49, train_loss:8.006253983694478e-06, test_loss:187.6459617614746
epoch: 54, train_loss:1.9808790739261896e-06, test_loss:187.6470153808594
epoch: 59, train_loss:4.904808179162501e-07, test_loss:187.6474952697754
epoch: 64, train_loss:1.2295471520928914e-07, test_loss:187.6476852416992
epoch: 69, train_loss:3.0720425314711974e-08, test_loss:187.6477961730957
epoch: 74, train_loss:7.80687452461848e-09, test_loss:187.6478421020508
epoch: 79, train_loss:1.9697075970270817e-09, test_loss:187.64787384033204
epoch: 84, train_loss:4.974581968575364e-10, test_loss:187.64787490844728
epoch: 89, train_loss:1.3226119435771898e-10, test_loss:187.64787322998046
epoch: 94, train_loss:4.4386938569118684e-11, test_loss:187.64787689208984
epoch: 99, train_loss:2.407660806835743e-11, test_loss:187.64787200927734
w的L2范数： 12.79533863067627
------权重惩罚=3------
epoch: 4, train_loss:9.545367622375489, test_loss:105.10482200622559
epoch: 9, train_loss:1.032323920726776, test_loss:73.30066757202148
epoch: 14, train_loss:0.16066254675388336, test_loss:51.89686546325684
epoch: 19, train_loss:0.049494955129921435, test_loss:37.018984375
epoch: 24, train_loss:0.03135346136987209, test_loss:26.606707515716554
epoch: 29, train_loss:0.0271341897547245, test_loss:19.30196159362793
epoch: 34, train_loss:0.025434764102101326, test_loss:14.16158353805542
epoch: 39, train_loss:0.024195411428809165, test_loss:10.529652681350708
epoch: 44, train_loss:0.02353227064013481, test_loss:7.9608730125427245
epoch: 49, train_loss:0.022784174606204032, test_loss:6.131768321990966
epoch: 54, train_loss:0.022214514762163164, test_loss:4.825783882141113
epoch: 59, train_loss:0.021587001159787178, test_loss:3.8851707649230955
epoch: 64, train_loss:0.020812243968248368, test_loss:3.2054775524139405
epoch: 69, train_loss:0.020173274911940098, test_loss:2.7079290342330933
epoch: 74, train_loss:0.019705117866396903, test_loss:2.3401026821136472
epoch: 79, train_loss:0.01901315413415432, test_loss:2.064272713661194
epoch: 84, train_loss:0.01864688694477081, test_loss:1.8552512741088867
epoch: 89, train_loss:0.017900681123137474, test_loss:1.692428708076477
epoch: 94, train_loss:0.01742161698639393, test_loss:1.5640966510772705
epoch: 99, train_loss:0.017075800523161887, test_loss:1.4593466567993163
w的L2范数： 0.47902122139930725
'''



