import numpy as np
import tensorflow as tf

"""
线性回归
用tensorflow上层API实现seqModel_basicCode.py的内容
#   1.生成带噪声的数据: x.shape=(1000,2); y.shape=(1000,1)
#   2.创建小批次数据迭代器
#   3.创建模型
#   4.初始化模型参数
#   5.创建损失函数
#   6.创建优化函数: SGD-> param-=lr*grad/batch_size
#   7.训练过程
"""

#   1.生成带噪声的数据
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = tf.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))
    return X, y

#   2.创建小批次数据迭代器
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个TensorFlow数据迭代器"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset

#已知模型架构: y=Xw+b
#创建带噪声的数据: y=Xw+b+噪声
true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10
data_iter = load_array((features, labels), batch_size)

#   3.创建模型
'''
y=Xw+b是单层网络架构, 这一单层被称为全连接层(fully-connected layer), 因为它的每一个输入都通过矩阵-向量乘法得到它的每个输出。
Dense:
    output=activation(X⋅W+b)
'''
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1)) #输出维度=1,activation=none

#   4.初始化模型参数
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))

#   5.创建损失函数
loss = tf.keras.losses.MeanSquaredError()

#   6.创建优化函数: SGD-> param-=lr*grad/batch_size
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)

#   7.训练过程
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

#与实际模型的参数误差
'''
w的估计误差： tf.Tensor([ 0.00034475 -0.00046492], shape=(2,), dtype=float32)
b的估计误差： [0.0004797]
'''
w = net.get_weights()[0]
print('w的估计误差：', true_w - tf.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('b的估计误差：', true_b - b)