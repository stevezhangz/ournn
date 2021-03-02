# Our neural network(Ournn) (V.0.01)
### Author: Steve Zhang Z    
### E_mail: stevezhangz@163.com 


## 中文介绍：
### 定位
Ournn 是我利用空闲时间独立开发的一款基于numpy的深度学习框架，Ournn意在于为对深度学习框架逻辑实现有需求的初学者提供启发，同时Ournn的最终目标是可以让大部分numpy使用者可以便捷地进行深度学习，而不必为熟悉其他格式的深度学习框架而重新学习新的编程。Ournn是一款开源免费框架，目前仍处于开发早期，但是已经初具规模。我欢迎对此项目感兴趣的朋友加入进来，人越多，这个项目实现的概率就越大。即便可能有些人已经尝试过类似的工作，但是我相信亲身参与到一些底层并且基础的项目，对于自己的耐心以及逻辑思维和意志的提升都有莫大的好处。同时，如果我们可以真正地让这款框架具备实际应用价值，我们的工作也就得到了认可。

### 现状
目前，Ournn已经兼容了最基本的全连接以及卷积层训练，具备权重保存、维度可视化、训练效果可视化等功能。但是由于本项目是独立开发，故还有很多常规函数以及功能没有落实，如果没有特殊情况，更新会持续;Ournn的优势在于numpy用户可以根据自己的想法去修改一些底层的参数，而不被条条框框所束缚住。比如，如果你使用tensorflow，你需要使用tensorflow中的一些API才可以访问或者修改tensorflow中的一些参数。而如今，基于numpy的ournn可以让他们有更广阔地操作空间。同时，我们希望此框架有更加灵活的训练及可视化功能，满足懒癌患者的一切需求，所以后续更新的功能才是重中之重;本框架有一定局限性。由于本框架基于python开发，因此无需考虑内存的配置，但是线程优化上存在劣势，同时numpy无法进行gpu加的，所以本框架也无法通过gpu进行加速，后续如果有时间，会尝试开发同numpy API类似的能够基于GPU加速计算数据包。

### 如何使用
step1: 将ournn这个package下载。并且放置到编译器的路径中。   

step2：具体的使用可以参考 test.py    

PS：可以根据自己的想象力去实现新的功能。    


## English Introduction:

### Positioning

Ournn Ournn is a deep learning framework based on numpy that I independently developed in my spare time. Ournn aims to provide inspiration for beginners who need to realize the logic of deep learning framework. At the same time, the ultimate goal of ournn is to enable most numpy users to easily carry out deep learning, without having to relearn NEW programming to be familiar with other deep learning frameworks. Ournn is an open source free framework, which is still in the early stage of development, but has begun to take shape. I welcome friends who are interested in this project to join us. The more people there are, the greater the probability of this project being realized. Even though some people may have tried similar work, I believe that personal participation in some low-level and basic projects is of great benefit to their patience, logical thinking and will. At the same time, if we can really make this framework have practical application value, our work will be recognized.



### Status quo

At present, ournn has been compatible with the most basic full connection and convolution layer training, with the functions of weight saving, dimension visualization and training effect visualization. However, due to the independent development of this project, there are still many conventional functions and functions that have not been implemented. If there is no special case, the update will continue. The advantage of ournn is that numpy users can modify some underlying parameters according to their own ideas without being bound by rules. For example, if you use tensorflow, you need to use some APIs in tensorflow to access or modify some parameters in tensorflow. Now, the ournn based on numpy can give them more space to operate. At the same time, we hope that this framework has more flexible training and visualization functions to meet all the needs of lazy cancer patients, so the follow-up update function is the top priority; this framework has some limitations. Because this framework is based on Python development, there is no need to consider the memory configuration, but there are disadvantages in thread optimization. At the same time, numpy can't add GPU, so this framework can't speed up through GPU. Later, if you have time, you will try to develop a GPU based data package similar to numpy API.



### How to use

Step 1: download the ournn package. And put it in the compiler's path.

Step 2: the specific use can be referred to test.py

PS: you can realize new functions according to your imagination.
