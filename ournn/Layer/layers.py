from ournn.Layer.activations import *
from ournn.tools.matrix_tools import *
from ournn.Layer.activations import *
from ournn.frame import *
from ournn.tools.matrix_tools import broadcast_bac
import numpy as np

class Conv2d:
    def __init__(self,
                 kernal_size,
                 channel_in,
                 channel_o,
                 act="relu",
                 us_bias=True,
                 stride=1,
                 padding=True,
                 name=None,
                 lambda_=None):
        self.lambda_=lambda_
        self.use_bias=us_bias
        self.conv_bias=None
        self.stride=stride
        if name:
            self.name=name
        else:
            self.name="conv"
        assert len(kernal_size)==2
        self.conv_filter=np.random.uniform(-1,1,(kernal_size[0],kernal_size[1],channel_in,channel_o))
        self.padding=padding
        self.sigmoid = sigmoid()
        self.relu = relu()
        self.actvation = {
            "relu": self.relu,
            "sigmoid": self.sigmoid
        }
        if act in self.actvation.keys():
            self.act = self.actvation[act]
        else:
            self.act = None

    def conv(self,image_set,conv_filter,stride):
        self.in_imageset = image_set
        out=[]
        index=[]
        self.act_sets=[]
        for img in image_set:
            conv_out,conv_index=self.conv_slid(img,conv_filter,stride)
            out.append(conv_out)
            index.append(conv_index)

            if self.act:
                line=[]
                for i in range(conv_out.shape[-1]):
                    conv_out[::,i]=self.act(conv_out[::,i])
                    line.append(self.act)
                self.act_sets.append(line)
        if self.act:
            self.act_sets=np.squeeze(np.array(self.act_sets))



        self.original_shape=image_set.shape
        self.output=np.array(out)
        self.index=np.array(index)
        return self.output

    def conv_slid(self,image,conv_filter,stride):


        # zeropadding
        length = ((image.shape[0] - conv_filter.shape[0]) // stride,
                  (image.shape[1] -  conv_filter.shape[1]) // stride,)
        if self.padding:
            image,length=zero_padding_same(image, conv_filter, stride)
        assert len(length)==2
        start_point=0
        end_point_x=image.shape[0]- conv_filter.shape[0]
        end_point_y=image.shape[1]- conv_filter.shape[1]
        new_img_c=np.zeros((length[0],length[1], conv_filter.shape[-1]))
        conv_index_record=np.zeros((length[0],length[1], conv_filter.shape[-1]))
        # convolution calculate
        for ch in range(image.shape[-1]):
            row_i=0
            for step_row in np.arange(start_point,end_point_x, stride):
                line_r=0
                row_i+=1
                for step_line in np.arange(start_point, end_point_y, stride):
                    line_r+=1
                    for o_ch in range( conv_filter.shape[-1]):
                        try:
                            new_img_c[row_i][line_r][o_ch] = self.conv_cal(image, ch, step_row, step_line, o_ch)
                            conv_index_record[row_i][line_r][o_ch] =(step_row, step_line,ch)
                        except:
                            continue
        self.usebias=0
        if self.use_bias:
            if self.usebias==0:
                self.conv_bias = np.random.uniform(-1, 1, size=new_img_c.shape)
                self.usebias+=1
            new_img_c+=self.conv_bias

        return new_img_c,conv_index_record

    def conv_cal(self,
                 img,
                 in_ch,
                 row,
                 line,
                 o_ch):

        filter_curr=self.conv_filter[:,:,in_ch,o_ch]
        pixel_curr=img[row:row+self.conv_filter.shape[0],line:line+self.conv_filter.shape[1],in_ch]
        return dot_mul2d(filter_curr,pixel_curr,self.conv_bias[in_ch][o_ch])


    def backward(self,err):


        rot_filter=self.conv_filter[::-1,::-1].reshape(self.conv_filter.shape[0],self.conv_filter.shape[1],
                                                                 self.conv_filter.shape[3],self.conv_filter.shape[2])
        assert err.shape==self.output.shape

        if self.use_bias:
            self.conv_bias-=np.mean(err,axis=0)
            if self.lambda_:
                self.conv_bias -=self.lambda_*self.conv_bias

        bac=np.zeros(self.original_shape)
        back_value=self.conv(err,rot_filter,self.stride)
        start_point = 0
        end_point_x = self.original_shape[1] - self.conv_filter.shape[0]
        end_point_y = self.original_shape[2] - self.conv_filter.shape[1]
        for im_i in range(bac.shape[0]):
            for ch in range(back_value.shape[-1]):
                mini_x=0
                for i_x in np.arange(start_point,end_point_x,self.stride):
                    mini_x+=1
                    mini_y=0
                    for i_y in np.arange(start_point,end_point_y,self.stride):
                        mini_y+=1
                        for i in range(self.conv_filter.shape[0]):
                            for j in range(self.conv_filter.shape[1]):
                                bac[im_i,i_x+i,i_y+j,ch]=back_value[im_i,mini_x,mini_y,ch]
                                self.conv_filter[i,j,ch]-=back_value[im_i,mini_x,mini_y,ch]*self.in_imageset[im_i,i_x+i,i_y+j,ch]
                                if self.lambda_:
                                    self.conv_filter[i,j,ch]-=self.lambda_*self.conv_filter[i,j,ch]
        return bac

    def return_variables(self):
        variables={}
        variables[self.name+"con_filter"]=self.conv_filter
        variables[self.name+"conv_bias"]=self.conv_bias
        return variables
    def load_variables(self,variables):
        self.conv_filter=variables[self.name + "con_filter"]
        self.conv_bias=variables[self.name + "conv_bias"]
    def __call__(self, image):
        self.x=image
        assert len(image.shape)==4
        self.y=self.conv(image,self.conv_filter,self.stride)
        return self.y+1e-5


class Maxpooling2d:

    def __init__(self,pooling_size,padding=True,stride=1,name=None):
        super().__init__()
        assert len(pooling_size)==2
        self.pool_size=pooling_size
        self.padding=padding
        self.stride=stride
        if name:
            self.name = name
        else:
            self.name="max_pool"

    def pool(self,imageset):
        assert len(imageset.shape)==4
        self.origin_shape=imageset.shape
        pool_output=[]
        self.max_index=[]
        for i in imageset:
            out,index=self.pool_slide(i)
            pool_output.append(out)
            self.max_index.append(index)
        return np.array(pool_output)

    def pool_slide(self,image):
        if self.padding:
            image, length = zero_padding_same(image, np.zeros(shape=(self.pool_size[0],self.pool_size[1])), self.stride)
        else:
            length = ((image.shape[0] - self.pool_size[0]) // self.stride,
                      (image.shape[1] - self.pool_size[1]) // self.stride,)
        pool_out=np.zeros(shape=(length[0],length[1],image.shape[-1]))
        max_index={}
        start_point=0
        end_point_x = image.shape[0] - self.pool_size[0]
        end_point_y = image.shape[1] - self.pool_size[1]
        # convolution calculate
        for ch in range(image.shape[-1]):
            row_i = 0
            for step_row in np.arange(start_point, end_point_x, self.stride):
                line_r = 0
                row_i += 1
                for step_line in np.arange(start_point, end_point_y, self.stride):
                    line_r += 1
                    try:
                        pool_out[row_i][line_r][ch],x,y = self.pool_cal(image, ch, step_row, step_line)
                        max_index[(row_i-1,line_r-1,ch)]=(x+step_row,y+step_line,ch)
                    except:
                        continue
        return pool_out,max_index

    def pool_cal(self,img, ch, row, line):
        pixel_curr=img[row:row+self.pool_size[0],line:line+self.pool_size[1],ch]
        max_= -1e8
        for i in range(pixel_curr.shape[0]):
            for j in range(pixel_curr.shape[1]):
                if pixel_curr[i][j]>max_:
                    max_=pixel_curr[i][j]
                    x=i
                    y=j

        return max_,x,y

    def __call__(self, imageset):

        return self.pool(imageset)

    def backward(self,bac):
        back_set=np.zeros(self.origin_shape)
        for i in range(bac.shape[0]):
            for j in range(bac.shape[1]):
                for k in range(bac.shape[2]):
                    for z in range(bac.shape[3]):
                        try:
                            index=self.max_index[i][(j,k,z)]
                            back_set[i,index[0],index[1],index[2]]=bac[i,j,k,z]
                        except:
                            continue
        return back_set
    def return_variables(self):
        return 0
    def load_variables(self, variables):
         pass


class Avgpooling2d:

    def __init__(self, pooling_size, padding=True, stride=1,name=None):
        super().__init__()
        assert len(pooling_size) == 2
        self.pool_size = pooling_size
        self.padding = padding
        self.stride = stride
        if name:
            self.name=name
        else:
            self.name="max_pool"


    def pool(self, imageset):
        assert len(imageset.shape) == 4
        pool_output = []
        for i in imageset:
            pool_output.append(self.pool_slide(i))
        return np.array(pool_output)

    def pool_slide(self, image):
        self.original_shape=image.shape
        if self.padding:
            image, length = zero_padding_same(image, np.zeros(shape=(self.pool_size[0], self.pool_size[1])), self.stride)
        else:
            length = ((image.shape[0] - self.pool_size[0]) // self.stride,
                      (image.shape[1] - self.pool_size[1]) // self.stride,)
        pool_out = np.zeros(shape=(length[0], length[1], image.shape[-1]))
        self.mean_record = np.zeros(shape=(length[0], length[1], image.shape[-1]))
        start_point = 0
        end_point_x = image.shape[0] - self.pool_size[0]
        end_point_y = image.shape[1] - self.pool_size[1]
        # convolution calculate
        for ch in range(image.shape[-1]):
            row_i = 0
            for step_row in np.arange(start_point, end_point_x, self.stride):
                line_r = 0
                row_i += 1
                for step_line in np.arange(start_point, end_point_y, self.stride):
                    line_r += 1
                    try:
                        pool_out[row_i][line_r][ch] = self.pool_cal(image, ch, step_row, step_line)
                        self.mean_record[row_i][line_r][ch]=(step_row, step_line)
                    except:
                        continue
        self.pool_out_shape=pool_out.shape
        return pool_out

    def pool_cal(self, img, ch, row, line):
        pixel_curr = img[row:row + self.pool_size[0], line:line + self.pool_size[1], ch]
        return np.mean(pixel_curr)

    def __call__(self, imageset):
        self.Pool_out=self.pool(imageset)
        return self.Pool_out

    def back_foward(self,bac):
        if self.pool_out_shape!=bac[0].shape:
            bac=bac.reshape(bac.shape[0],self.pool_out_shape[0],self.pool_out_shape[1],self.pool_out_shape[2])
        back_arr=np.zeros(shape=self.original_shape)
        for i in range(self.mean_record.shape[-1]):
            for j in range(self.mean_record.shape[0]):
                for k in range(self.mean_record.shape[1]):
                    if self.mean_record[j][k][i]!=0:
                        back_arr[self.mean_record[j][k][i][0]:self.mean_record[j][k][i][0]+self.pool_size[0],
                        self.mean_record[j][k][i][1]:self.mean_record[j][k][i][1]+self.pool_size[1],
                        i]=+bac[j][k][i]

        assert back_arr.shape==self.original_shape
        return back_arr
    def return_variables(self):
        return 0
    def load_variables(self, variables):
         pass






class Fully_connected:
    def __init__(self,output_dim,input_dim=None,use_bias=True,name=None,act=None,local_trainable=True,lambda_=0.01):
        self.name=name
        if not self.name:
            self.name = "fc_"
        self.use_bias=use_bias
        self.output_dim=output_dim
        self.input_dim = input_dim
        if self.input_dim!=None:
            self.weight=np.random.uniform(-1,1,(self.input_dim,self.output_dim))
            self.bias = np.random.uniform(-1, 1, (1, self.output_dim))
        self.local_trainable=local_trainable
        self.relu=relu()
        self.sigmoid=sigmoid()
        self.softmax=softmax()
        self.actvation={
            "relu":self.relu,
            "sigmoid":self.sigmoid,
            "softmax":self.softmax
        }

        if act==None:
            self.act=act

        else:
            if act in self.actvation.keys():
                self.act=self.actvation[act]
        self.lambda_=lambda_
    def forward(self,x):
        self.x = x
        if not isinstance(x,np.ndarray):
            x=np.array(x)
        if self.use_bias:
            y=np.dot(x,self.weight)+self.bias
        else:
            y = np.dot(x, self.weight)
        if self.act:
            self.y=self.act(y)
            return self.y
        else:
            self.y=y
            return y

    def variables(self,index=0):
        if self.input_dim==None:
            self.input_dim = x.shape[-1]
            self.weight = np.random.uniform(-1, 1, (self.input_dim, self.output_dim))
            self.bias = np.random.uniform(-1, 1, (1, self.output_dim))
        #super(skeleton, self).__init__(index)
        if self.name==None:
            name="fc"
        self.name = self.name + "{}".format(index)
        print(self.name+"'s variable list:\n")
        print(self.name+"'s bias\n")
        print(self.bias)
        print(self.name+"'s weight\n")
        print(self.weight)
        return self.weight,self.bias

    def backward(self,backvalue):
        if not self.act:
            backvalue=broadcast_bac(backvalue,self.y)
        else:
            backvalue=self.act.backward(backvalue)
        self.dbias = np.mean(backvalue,axis=0)
        self.dw = np.dot(self.x.T, backvalue)
        if self.lambda_:
            self.dw+=self.lambda_*self.weight
            self.dbias=self.dbias.reshape((1,-1))
            self.dbias+=self.lambda_*self.bias
        weight_ = self.weight
        if self.local_trainable==True:
            self.weight-=self.dw
            if self.use_bias:
                self.bias-=self.dbias
        return np.dot(backvalue,weight_.T)

    def return_variables(self):
        variables = {}
        variables[self.name + "fc_weight"] = self.weight
        variables[self.name + "fc_bias"] = self.bias
        return variables

    def load_variables(self, variables):
         self.weight=variables[self.name + "fc_weight"]
         self.bias=variables[self.name + "fc_bias"]
    def __call__(self,x):
        if self.input_dim==None:
            self.input_dim = x.shape[-1]
            self.weight = np.random.uniform(-1, 1, (self.input_dim, self.output_dim))
            self.bias = np.random.uniform(-1, 1, (1, self.output_dim))
        return self.forward(x)

















class Flatten:
    def __init__(self,name="Flatten_"):
        self.name=name
    def __call__(self,x):
        return self.forward(x)
    def forward(self,x):
        self.x = x
        if not isinstance(x, np.ndarray):
            self.x = np.array(self.x)
        self.y = self.x.reshape(self.x.shape[0], -1)
        return self.y
    def return_variables(self):
        return 0
    def load_variables(self, variables):
         pass
    def backward(self,bac):
        return bac.reshape(self.x.shape)
