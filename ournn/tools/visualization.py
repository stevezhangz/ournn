import numpy as np
""
def train_process_visualization(i,epoches,err,acc,val_err=None,val_acc=None,other_info=None):
    info="Download progress: {}%: ".format(int(((i + 1) / epoches) * 100))+"▋" * int((i / epoches) * 10)+\
             "epoches:{}---err:{}---acc:{}".format(i + 1, err, acc)
    if val_err :
        info = info+"---val_err:{}".format(val_err)
    if val_acc:
        info = info + "---val_acc:{}".format(val_acc)
    if other_info:
        assert isinstance(other_info,dict)
        for key in other_info.keys():
            info+="---"+key+":{}".format(other_info[key])
    info+=""
    return info