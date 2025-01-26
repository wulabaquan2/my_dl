import random
import tensorflow as tf
"""
不用上层API实现以下内容:
#   1.生成带噪声的数据
#   2.创建小批次数据迭代器
#   3.创建模型
#   4.初始化模型参数
#   5.创建损失函数
#   6.创建优化函数: SGD-> param-=lr*grad/batch_size
#   7.训练过程
"""

#生成带噪声的数据
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = tf.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))
    return X, y

#创建小批次数据迭代器
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)

#创建模型
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return tf.matmul(X, w) + b

#创建损失函数
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

#创建优化函数: SGD-> param-=lr*grad/batch_size
def sgd(params, grads, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)


#已知模型架构: y=Xw+b
#创建带噪声的数据: y=Xw+b+噪声
true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10

#初始化模型的参数
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)

#设置超参数
lr = 0.03   #学习率
num_epochs = 3  #完整数据迭代周期数
net = linreg    #模型别名为net
loss = squared_loss #损失函数别名为loss

#训练过程
for epoch in range(num_epochs): 
    for X, y in data_iter(batch_size, features, labels):    #每个小批次的数据X,y
        with tf.GradientTape() as g:        
            l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 计算l关于[w,b]的梯度
        dw, db = g.gradient(l, [w, b])  
        # 使用参数的梯度更新参数
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels) #计算当前epoch优化后的参数的损失
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}') #输出平均损失

#与实际模型的参数误差
'''
运行结果:
w的估计误差: [ 0.00035858 -0.00064158]
b的估计误差: [0.00047207]
'''
print(f'w的估计误差: {true_w - tf.reshape(w,true_w.shape)}')
print(f'b的估计误差: {true_b - b}')