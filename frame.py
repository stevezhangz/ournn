import numpy as np
import matplotlib.pyplot as plt
import time
import sys
class skeleton:

    def __init__(self, name=None,Regularization=None):
        self.container = []
        self.name = name
        self.lambda_=Regularization
    def add(self, functions):
        try:
            assert len(functions) >= 1
            count = 0
            for i in functions:
                count += 1
                i.name += str(count)
                self.container.append(i)
        except:
            self.container.append(functions)
        self.bac_container = self.container[::-1]
        if self.lambda_:
            for i in range(len(self.container)):
                self.container[i].lambda_=self.lambda_
    def __add__(self, other_model):
        for layer in other_model.container:
            self.container.append(layer)
    def train(self, x, y,
              lr=None,
              early_stop=10,
              epoches=100,
              optimizer=None,
              val_data=None,
              train_test_split=None,
              shufflue=False,
              call_back=None):

        self.epoch = epoches
        history = {"acc": [],
                   "loss": []}
        # if u didn't select an optimizer, this code will fellow the ordinary setting.
        if not optimizer:
            if not lr:
                lr = 1e-3
            count = 0
            reco1 = 0
            reco2 = 0
            for i in range(epoches):
                y_pre = self.predict(x)
                if i < 1:
                    err = np.mean(np.matmul((y_pre - y), (y_pre - y).T))
                if i > 1:
                    err_new = np.mean(np.matmul((y_pre - y), (y_pre - y).T))
                    if ((err_new - err) / (err + 1e-3) > 10):
                        lr *= 0.1 * 1 / ((err_new - err) / (err + 1e-3))
                    elif ((err - err_new) / (err + 1e-3) < 0.1):
                        lr *= 1.1
                        reco1 = i
                        if reco1 != 0:
                            reco2 = i
                        if reco1 == (reco2 - 1):
                            count += 1
                        else:
                            reco1 = i
                            reco2 = 0
                            count = 0
                        count += 1
                    if count >= early_stop or ((err_new - err) / err > 0.3) and acc < 1.5:
                        self.history = history
                        return history
                    err = err_new
                self.backward(err * lr)
                acc = np.mean(np.abs((y_pre - y) / (y + 1e-3)))
                history["acc"].append(acc)
                history["loss"].append(err)
                print("\r", end="")
                print("Download progress: {}%: ".format(int(( (i+1)/ epoches) * 100)), "▋" * int((i / epoches) * 10),
                      "epoches:{}---err:{}---acc:{}".format(i+1, err, acc), end="")
                sys.stdout.flush()
                time.sleep(0.01)

            self.history = history
            return history

        else:
            self.container,back,self.history=optimizer(x,
                                                            y,
                                                            layers=self.container,
                                                            epoches=epoches,
                                                            train_test_split_size=train_test_split,
                                                            call_back=call_back)
        return self.history
    def backward(self, bac):
        for i in self.bac_container:
            bac = i.backward(bac)

    def predict(self, datahead):
        for layer_forward in self.container:
            datahead = layer_forward(datahead)
        return datahead

    def save_weights(self,path):
        import os
        import json
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path,"w") as f:
            variables=[]
            for i in range(len(self.container)):
                variables.append(self.container[i].return_variables)
            json.dump(variables,f)



    def load_weights(self,path):
        import os
        import json
        if not os.path.exists(path):
            raise FileNotFoundError
        else:
            with open(path,"w") as f:
                variables=json.load(f)
            for i in range(len(self.container)):
                if variables[i]!=0:
                    self.container[i].load_variables(variables[i])



    def show_info(self):
        print("\n")
        print("----------------------------------------------------------------------------------------")
        if not self.name:
            print("name" + "'s info")
        else:
            print("model's info are shown as below:")
        print("----------------------------------------------------------------------------------------")
        count = 1
        param_count = 0
        for i in self.container:
            count += 1
            c = 1
            pre_count = param_count
            try:
                for c in i.weight.shape:
                    c *= c
                param_count += c
                param_count += i.bias.shape[0]
                for c in i.conv_filter.shape:
                    c *= c
                param_count += c
                param_count += i.conv_bias.shape[0]
                print("\t" + i.name + "'s variable:---in_dim:", i.x.shape, "---out_dim:", i.y.shape,"---params:{}".format(param_count-pre_count))
                print("\t" + i.name + "'s variable:---in_dim:", i.x.shape, "---out_dim:", i.y.shape,
                      "---params:{}".format(param_count - pre_count))
            except:
                print("\t" + i.name + "'s variable:---in_dim:", i.x.shape, "---out_dim:", i.y.shape,
                      "---params:{}".format(param_count - pre_count))
                continue
        print("----------------------------------------------------------------------------------------")
        print("total {} parameters".format(param_count))
        print("----------------------------------------------------------------------------------------")


    def visualization(self, name="output"):
        if self.history:
            plt.figure(figsize=(10, 10))
            plt.grid()
            for i in self.history.keys():
                plt.plot(self.history[i])
            plt.legend()
            plt.savefig(name + ".jpg")
            plt.show()
