import numpy as np
from ournn.tools.matrix_tools import dot_mul2d
import math



class MSE:
    def __init__(self):
        pass
    def loss(self,x,y,delta=1e-3):
        if isinstance(x,int) or isinstance(x,float):
            if isinstance(y, int) or isinstance(y, float):
                return (x-y)*(x-y)
        assert x.shape==y.shape
        add_=0
        self.err = np.square((x - y))
        return self.err

    def __call__(self,x,y):
        self.x=x
        self.y=y
        return self.loss(x,y)

    def backward(self):
        return 2*(self.x-self.y)

"""
Cross entropy
"""
class sparse_logit_cross_entropy:
    def __init__(self):
        pass
    def loss(self,x,y):
        if isinstance(x,int) or isinstance(x,float):
            if isinstance(y, int) or isinstance(y, float):
                return -y*np.log(x)
        x=x.reshape(y.shape)
        assert x.shape==y.shape
        out=-np.log(x)*y
        return out

    def __call__(self, x,y):
        self.x=x
        self.y=y
        return self.loss(x,y)
    def backward(self):
        if isinstance(self.x,int) or isinstance(self.x,float):
            if isinstance(self.y, int) or isinstance(self.y, float):
                return self.y/(self.x)
        self.x=self.x.reshape(self.y.shape)
        cross_entropy=[]
        assert self.x.shape==self.y.shape
        out=-(1/(self.x))*self.y
        return out


"""
The predicted values were processed by softmax and then calculated by cross entropy
In another word the last layer of the Model dont have to use the softmax act function
"""
class sparse_softmax_cross_entropy:
    def __init__(self):
        pass
    def loss(self,x,y,logit=sparse_logit_cross_entropy(),down_delta=1e-3,upsume=1e5):
        self.x=x
        self.y=y
        if isinstance(x,int) or isinstance(x,float):
            raise FileExistsError
        assert x.shape==y.shape
        out=[]
        x+=1e-5
        for i in range(x.shape[0]):
            line_sotmax=[]
            line_sotmax.append((x[i,:]/(np.sum(x[i,:]))))
            out.append(line_sotmax)
        out=np.squeeze(np.array(out))
        cross_entropy_out=logit(out,y)
        self.logit=logit
        self.softout=out
        return cross_entropy_out
    def __call__(self,x,y):
        return self.loss(x,y)

    def backward(self):
        logit_back=self.logit.backward()
        exp_x_n=1/(np.exp(-(self.x))+1e-5)
        bac=self.softout*(-1+self.softout/exp_x_n)*logit_back
        return bac