import numpy as np

def sparse_one_hot_encode(y):
    if not isinstance(y,np.ndarray):
        y=np.array(y)
    assert len(y.shape)==2
    unique=np.unique(y)
    new=[]
    for i in range(y.shape[0]):
        line=np.arange(0,len(unique),1)*0
        for j in range(y.shape[1]):
            index=np.where(unique==y[i][j])[0][0]
            line[index]=1
        new.append(line)
    new=np.squeeze(np.array(new))
    return new



def binary_one_hot_encode(y):

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    for iy in range(y.shape[1]):
        seq_a = np.unique(y[:, iy])
        for ix in range(y.shape[0]):
            if y[ix,iy]==seq_a[0]:
                y[ix, iy] =1
            if y[ix,iy]==seq_a[1]:
                y[ix, iy] = 0
    return y

def train_test_split(train_x,train_y,random_shuffle=True,split_size=None,batch_size=None):
    assert train_x.shape[0]==train_y.shape[0]
    index=np.arange(0,train_x.shape[0],1)

    if random_shuffle:
        np.random.shuffle(index)

    if batch_size==None:
        batch_size=train_x.shape[0]

    if batch_size:

        index=index[:batch_size]

    v_x,v_y=None,None
    if split_size:
        split_index=index.shape[0]*split_size

        train_index=index[:int(split_index)]
        val_index=index[int(split_index):]

        t_x,t_y=train_x[train_index],train_y[train_index]
        v_x,v_y=train_x[val_index],train_y[val_index]
    else:
        t_x,t_y=train_x,train_y

    if isinstance(v_x,np.ndarray):
            return t_x, t_y, v_x, v_y
    else:
        return t_x, t_y



