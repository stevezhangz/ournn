import numpy as np
from tensorflow.keras.datasets import mnist
from ournn.tools.preprocess import sparse_one_hot_encode
from ournn.tools.matrix_tools import *
from ournn.frame import skeleton
from ournn.Layer.layers import *
from ournn.optimizers import *
from ournn.losses import *

#无奈地拿起了tensorflow的数据
(x,y),(t,d)=mnist.load_data()
x=np.expand_dims(x,axis=-1)
y=y.reshape(-1,1)
x,y=x[0:400],y[0:400]
x=(x-x.max())/(x.max()-x.min())
#热编码
y=sparse_one_hot_encode(y)
#初始化框架
sk=skeleton(name="Model1",Regularization=None)
#将不同的层添加到框架中
sk.add(
    [
    Conv2d(kernal_size=(5,5),padding=True,stride=2,channel_in=1,channel_o=3),
    Flatten(),
    Fully_connected( output_dim=500,act="relu"),
    Fully_connected( output_dim=100,act="relu"),
    Fully_connected(output_dim=10,act="relu")
    ]
)
#优化器
optimizer=SGD(loss=sparse_softmax_cross_entropy(),sample_size=0.7,lr=1e-5)
#训练
history=sk.train(x,y,epoches=20,train_test_split=0.7,optimizer=optimizer)
#显示维度信息
sk.show_info()
#将损失以及精度绘图
sk.visualization()
