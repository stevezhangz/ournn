import numpy as np
from ournn.tools.matrix_tools import broadcast_bac
"""
It is suitable for two-dimensional and one-dimensional calculation. The element greater than 0 returns the value itself, 
and the element less than 0 returns 0.
"""
class relu:
    def __init__(self):
        pass
    def forward(self,x):
        if isinstance(x,int) or isinstance(x,float):
            if x<0:
                x=0
            return x
        else:
            if not isinstance(x,np.ndarray):
                x=np.array(x)
            assert len(x.shape)==2
            for i in range(x.shape[0]):
                for j in  range(x.shape[1]):
                    if x[i][j]<0:
                        x[i][j]=0
        return x

    def __call__(self, x):
        self.y=self.forward(x)
        return self.y

    def backward(self,back_value):
        back_value=broadcast_bac(back_value,self.y)
        bac=np.zeros(shape=back_value.shape)
        if isinstance(back_value, int) or isinstance(back_value, float):
            if back_value>0:
                return 1
            else:
                return 0
        else:
            for i in range(back_value.shape[0]):
                for j in range(back_value.shape[1]):
                    if back_value[i][j] <=0:
                        bac[i][j] = 0
                    else:
                        bac[i][j] = 1

        back_value=bac*back_value
        return back_value

"""
It is suitable for two-dimensional and one-dimensional calculation and returns 1./(1.+exp(-x[i][j]))
"""
class sigmoid:
    def __init__(self):
        pass
    def forward(self, x,up_sume=None):
        if isinstance(x,int) or isinstance(x,float):
            return 1./(1.+np.exp(-x))
        else:
            if not isinstance(x,np.ndarray):
                x=np.array(x)
            assert len(x.shape)==2
            for i in range(x.shape[0]):
                for j in  range(x.shape[1]):
                    if up_sume:
                        x[i][j]=1./(1.+np.exp(-x[i][j]))
                    else:
                        x[i][j] = 1. / (1. + np.exp(-x[i][j]))
        return x
    def __call__(self, x):
        self.y=self.forward(x)
        return self.y
    def backward(self, back_value):
        back_value=broadcast_bac(back_value,self.y)
        if isinstance(back_value, int) or isinstance(back_value, float):
            return back_value*self.y*self.y*1./(1-self.y)
        else:
            for i in range(back_value.shape[0]):
                for j in range(back_value.shape[1]):
                    back_value[i][j]*=self.y[i][j]*self.y[i][j]*1./(1-self.y[i][j]+1e-3)
            return back_value

class softmax:
    def __init__(self):
        pass
    def forward(self, x,up_sume=None):
        if isinstance(x,int) or isinstance(x,float):
            raise ValueError
        else:
            if not isinstance(x,np.ndarray):
                x=np.array(x)
            assert len(x.shape)==2
            out = []

            for i in range(x.shape[0]):
                line_sotmax = []
                line_sotmax.append((x[i, :] / (1e-5 + np.sum(x[i, :]))))
                out.append(line_sotmax)
            out = np.squeeze(np.array(out))
        return out
    def __call__(self, x):
        self.x=x
        self.y=self.forward(x)
        return self.y
    def backward(self, back_value):
        back_value=broadcast_bac(back_value,self.y)
        if isinstance(back_value, int) or isinstance(back_value, float):
            raise ValueError
        else:
            exp_x_n = 1 / (1e-3 + np.exp(-self.x))
            bac = self.y * (-1 + self.y / exp_x_n) * back_value
            return bac