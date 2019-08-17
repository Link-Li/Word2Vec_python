# Word2Vec_python

**The Englist please refer to <a href="https://github.com/Link-Li/Word2Vec_python" target="_blank">README</a>**

经过一段时间的学习，开始准备写这些Python代码，目前还是更新中，写代码的时候主要参考了以下一些大佬的代码

**<a href="https://github.com/dav/word2vec" target="_blank">dav/word2vec</a>**

**<a href="https://github.com/liuwei1206/word2vec" target="_blank">liuwei1206/word2vec</a>**

**<a href="https://github.com/linshouyi/Word2VEC_java" target="_blank">linshouyi/Word2VEC_java</a>**

同时特别感谢写了**word2vec中的数学**的这位大佬peghoty@163.com，我也将这份pdf放到了项目里，大家可以进行学习参考

### cbow
目前已经完成了cbow的单线程代码,如果需要进行训练,则需要运行`Word2Vec_cbow_hs_1_one_thread.py`,参数设定都在代码中进行修改

我的cpu是i7-6700HQ,如果你是第一次运行代码,那么需要先设定save_vocab_file_name参数,这个参数是告诉代码将处理好的单词存储在哪.如果使用数据集`text8`,大概需要40分钟左右来查找需要训练的单词,然后存储到文件中.如果再次进行训练,直接读取提取好的单词的文件,那么速度就会很快.之后大概需要24个小时以上的训练,我测试了4次,训练时间从24小时到48小时不等.

目前完成了cbow的多进程代码,见代码`Word2Vec_cbow_hs_1_multithread.py`,使用参数`num_thread`来指定使用的进程个数,我在i7-9700K上面使用10个进程训练,大概速度在`3.5k/thread/s`,运行了大概2个小时左右,在i7-6700HQ,大概需要6个小时左右.这里必须吐槽一下python的多进程数据交互问题,速度非常慢,使用单线程运算的时候,,在i7-6700HQ上面,速度在`2.5k/thread/s`左右,但是使用多线程的时候,速度直接减少了一半.


你可以在这里下载到我处理好的词<a href="https://pan.baidu.com/s/1ruOs7RFy140L8L9UHvBKIw" traget="_blank">百度网盘</a>,提取码是:fs5t.其中`text8`是数据集,`text8_vocab_test_1000_5.txt`是我处理好的词,可以直接读取.

### distance
使用`Vector_Distance.py`代码进行距离的计算,其中需要在代码中指定训练好的词向量的文件位置

你可以在这里下载到我使用cbow训练好的词向量的模型<a href="https://pan.baidu.com/s/1ruOs7RFy140L8L9UHvBKIw" traget="_blank">百度网盘</a>,提取码是:fs5t.文件是`text8_vector_cbow_1.npz`.


