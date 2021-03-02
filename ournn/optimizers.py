import numpy as np
from ournn.frame import skeleton
from ournn.tools.visualization import train_process_visualization
from ournn.tools.preprocess import train_test_split
import sys
import time


class SGD:
    def __init__(self, loss,
                 epoches=None,
                 sample_size=0.5,
                 lr=None,
                 batch_size=None,
                 random_shuffle=True,
                 alpha=None,
                 v=None):
        self.sample_size=sample_size
        self.loss = loss
        self.shuffle = random_shuffle
        self.lr = lr
        self.batch_size = batch_size
        self.history = {
            "acc": [],
            "err": []
        }

    def optimizer(self, x, y, loss, layers, layers_bac, epoches=None, lr=None, sample_size=None, alpha=None, v=None):
        if isinstance(x, int) or isinstance(x, float):
            raise FileExistsError
        if lr==None:
            lr=1e-3
        assert x.shape[0] == y.shape[0]
        index = range(x.shape[0])
        select_size = sample_size * x.shape[0]
        index_select = np.random.choice(index, int(select_size))
        x=x[index_select]
        for layer in layers:
            x=layer(x)
        err =loss(x, y[index_select])


        back = loss.backward() * lr

        for layer in layers_bac:
            back = layer.backward(-back)
        acc=np.mean(np.abs(x-y[index_select])/(y[index_select]+1e-3))
        return acc,err, layers_bac,layers, back

    def __call__(self, x, y,layers,epoches, train_test_split_size, train_state=True, call_back=[]):
        def pforward(layers,val_x):
            for layer in layers:
                val_x=layer(val_x)
            return val_x
        if not isinstance(x, np.ndarray):
            train_x = np.array(x)
        if not isinstance(y, np.ndarray):
            train_y = np.array(y)
        assert x.shape[0] == y.shape[0]

        layer_bac=layers[::-1]

        if self.batch_size==None:
            self.batch_size=int(x.shape[0])
        if isinstance(self.batch_size,float):
            self.batch_size=int(self.batch_size)
        if self.batch_size<2:
            raise ValueError
        if train_test_split_size:
            self.history["val_acc"]=[]
            self.history["val_loss"]=[]


        for i in range(epoches):
            if train_test_split_size:
                train_x, train_y, val_x, val_y = train_test_split(x, y, random_shuffle=self.shuffle,
                                                                  split_size=train_test_split_size,
                                                                  batch_size=self.batch_size)
            else:
                train_x, train_y = train_test_split(x, y, random_shuffle=self.shuffle, batch_size=self.batch_size)
            acc,err, layer_bac,layers, back = self.optimizer(train_x,train_y,layers_bac=layer_bac, layers=layers,loss=self.loss,  lr=self.lr,
                                                     sample_size=self.sample_size)
            self.history["err"].append(np.mean(err))
            self.history["acc"].append(acc)
            if train_test_split_size:
                val_pre = pforward(layers,val_x)
                val_err = np.mean(self.loss(val_pre, val_y))
                val_acc=np.mean(np.abs(val_pre-val_y)/(val_y+1e-3))
                self.history["val_acc"].append(val_acc)
                self.history["val_loss"].append(val_err)
                info=train_process_visualization(i,epoches,np.mean(err),acc,val_err=val_err,val_acc=val_acc,other_info=None)
            else:
                info = train_process_visualization(i, epoches, np.mean(err), acc, val_err=None, val_acc=None,
                                            other_info=None)
            print("\r", end="")
            print(info,end="")
            sys.stdout.flush()
            time.sleep(0.05)
        print("\n")
        return layers, back,self.history


