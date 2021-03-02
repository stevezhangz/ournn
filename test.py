import numpy as np
from tensorflow.keras.datasets import mnist
from ournn.tools.preprocess import sparse_one_hot_encode
from ournn.tools.matrix_tools import *
from ournn.frame import skeleton
from ournn.Layer.layers import *
from ournn.optimizers import *
from ournn.losses import *


(x,y),(t,d)=mnist.load_data()
x=np.expand_dims(x,axis=-1)
y=y.reshape(-1,1)
x,y=x[0:400],y[0:400]
x=(x-x.max())/(x.max()-x.min())
y=sparse_one_hot_encode(y)
sk=skeleton(name="Model1",Regularization=None)
sk.add(
    [
    Conv2d(kernal_size=(5,5),padding=True,stride=2,channel_in=1,channel_o=3),
    Flatten(),
    Fully_connected( output_dim=500,act="relu"),
    Fully_connected( output_dim=100,act="relu"),
    Fully_connected(output_dim=10,act="relu")
    ]
)
optimizer=SGD(loss=sparse_softmax_cross_entropy(),sample_size=0.7,lr=1e-5)
history=sk.train(x,y,epoches=20,train_test_split=0.7,optimizer=optimizer)
out=sk.predict(x)
sk.show_info()
sk.visualization()