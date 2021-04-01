# Our neural network(Ournn) (V.0.01)
### Author: Steve Zhang Z    
### E_mail: stevezhangz@163.com 


## 关于Ournn：
1. Ournn 是为numpy量身定制的类深度学习框架，致力于让用户使用numpy格式的数据进行深度学习。其主要的目标如下:    
* 可训练
  * 能够进行最基本的深度学习训练是该框架最基本的目标。
* 性能上的提升
  * 我们将从个角度进行提升：1.开发GPU版本的numpy数据包。2.对框架中计算复杂度比较高的部分进行改动，同时努力将硬件因素考虑到框架设计过程中。 以上两点同时存在，不存在先后顺序。
* 可视化
  * 可视化是深度学习研究中必不可少的环节，我们致力于将部分重要的可视化环节封装到框架中，从而让用户避开收集数据、自行可视化的繁琐流程。
* 简洁简洁再简洁
  * 我们希望简单化搭建神经网络的过程，从而让人们不需要有太多框架使用经验就可快速上手，这是本项目的最终目标。
2. 为了实现以上目标，我们付出了一定精力，且目前已经有了些许进展：
* 常用函数
  * 我们已经在框架中封装了最基本的一些函数：1. 神经网络方面 全连接层、卷积层以及个别的的激活函数和损失函数。
* 搭建网络
  * 由于我们是该领域从事基础研究的学生，故有其他框架的使用经验。通过参考其他人的框架，以及结合个人的懒癌精神，给出了最方便的搭建方法，在下文中有详细地介绍。
3. 我们也清晰地认识到，该项目距离最终目标还有一段距离，所以接下来我们的重心在于：
* 结合性能上的考虑进一步优化整体的设计思路。
* 完善目前深度学习所需要的各类函数。


## 如何使用
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
