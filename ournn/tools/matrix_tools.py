import numpy as np
"""
Multiply and add the values of the same position in the matrix of the same dimension
"""
def dot_mul2d(arr,arr1,bias=None,average=False):
    assert arr.shape==arr1.shape
    add_=0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            add_+=arr[i][j]*arr1[i][j]
    if average:
        add_=add_/(arr.shape[0]*arr.shape[1])
    return add_

"""
In order to avoid the loss of the dimension information of the convoluted matrix, the function first needs to add 
several lines of 0 in the lower left foot of the matrix to retain the dimension information as much as possible, and 
then creates a zero matrix with the size of the convoluted matrix, and adds the convolution calculation value of the 
original image of the corresponding position to the theoretical non-zero position.
"""
def zero_padding_same(image,conv_filter,stride):
    count = 0
    count2= 0
    if ((image.shape[0] - conv_filter.shape[0] + count) //stride != (image.shape[0]) // stride or
        (image.shape[1] - conv_filter.shape[1] + count2) /stride != (image.shape[1]) / stride or stride == 1 or
            (image.shape[1] - conv_filter.shape[1]) %   stride != 0 or
            (image.shape[0] - conv_filter.shape[0]) %   stride != 0):
        while ((image.shape[0] - conv_filter.shape[0] + count) //stride != (image.shape[0]) // stride and
        (image.shape[1] - conv_filter.shape[1] + count2) /stride != (image.shape[1]) / stride):
            if (image.shape[0] - conv_filter.shape[0] + count) //stride != (image.shape[0]) //stride:
                count += 1
            if (image.shape[1] - conv_filter.shape[1] + count2) //stride != (image.shape[1]) //stride:
                count2 +=1
    new_side_length_x = image.shape[0] + count
    new_side_length_y = image.shape[0] + count2
    new_contain = np.zeros(shape=(new_side_length_x, new_side_length_y, image.shape[-1]))
    for i in range(image.shape[-1]):
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                new_contain[j][k][i] = image[j][k][i]
    image = new_contain
    length = ((new_side_length_x -conv_filter.shape[0]) // stride,(new_side_length_y -conv_filter.shape[1]) // stride)
    return image,length

"""
In the process of back propagation, the dimension of error and the dimension of data may not match, including the pheno-
menon that the error may be non scalar and equivalent to averaging the upper level data (numpy and tensor are different 
forms, everything in tensor is scalar, so tensor does not have to worry about non scalar data)
"""

def broadcast_bac(backvalue, pre):
    if isinstance(backvalue, int) or isinstance(backvalue, float):
        if isinstance(pre, int) or isinstance(pre, float):
            return backvalue
        else:
            backset = []
            for i in range(pre.shape[0]):
                line = []
                for j in range(pre.shape[1]):
                    line.append(backvalue)
                backset.append(line)
            del (backvalue)
            backvalue = np.squeeze(np.array(backset)).reshape(pre.shape[0], -1)
            assert backvalue.shape == pre.shape
            return backvalue
    else:
        backvalue=np.array(backvalue).reshape(backvalue.shape[0],-1)
        assert backvalue.shape[0]==pre.shape[0]
        if pre.shape[1]==1:
            return backvalue
        else:
            bac=[]
            for index,val in enumerate(backvalue):
                line=[]
                for num in range(pre.shape[1]):
                    line.append(val)
                bac.append(val)

            backvalue=np.squeeze(np.array(bac)).reshape(pre.shape)
        assert backvalue.shape == pre.shape
        return backvalue

































"""
    if backvalue.shape == pre.shape:
        assert backvalue.shape == pre.shape
        return backvalue
    else:
        if backvalue.shape[0] == pre.shape[0]:
            if backvalue.shape[-1] == pre.shape[-1]:
                assert backvalue.shape == pre.shape
                return backvalue
            else:
                back_set = []
                for i in range(pre.shape[0]):
                    line = []
                    for j in range(pre.shape[1]):
                        line.append(backvalue[i])
                    back_set.append(line)
                backvalue = np.squeeze(np.array(back_set))

                assert backvalue.shape == pre.shape
                return backvalue
        else:
            if backvalue.shape[1] == pre.shape[1]:
                back_set = []
                for i in range(pre.shape[0]):
                    back_set.append(backvalue)
                backvalue = np.squeeze(np.array(back_set))
                return backvalue
    print(backvalue.shape,pre.shape)
    assert backvalue.shape == pre.shape
    return backvalue
"""