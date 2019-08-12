# Word2Vec_python

**The Englist please refer to <a href="https://github.com/Link-Li/Word2Vec_python" target="_blank">README</a>**

经过一段时间的学习，开始准备写这些Python代码，目前还是更新中，写代码的时候主要参考了以下一些大佬的代码

**<a href="https://github.com/dav/word2vec" target="_blank">dav/word2vec</a>**

**<a href="https://github.com/liuwei1206/word2vec" target="_blank">liuwei1206/word2vec</a>**

**<a href="https://github.com/linshouyi/Word2VEC_java" target="_blank">linshouyi/Word2VEC_java</a>**

同时特别感谢写了**word2vec中的数学**的这位大佬peghoty@163.com，我也将这份pdf放到了项目里，大家可以进行学习参考

### cbow
目前已经完成了cbow的单线程代码,如果需要进行训练,则需要运行`Word2Vec_cbow_hs_1_one_thread.py`,参数设定都在代码中进行修改

我的cpu是i7-6700HQ,如果你是第一次运行代码,那么需要先设定save_vocab_file_name参数,这个参数是告诉代码将处理好的单词存储在哪.如果使用数据集`text8`,大概需要40分钟左右来查找需要训练的单词,然后存储到文件中.如果再次进行训练,直接读取提取好的单词的文件,那么速度就会很快.

之后大概需要24个小时以上的训练,我测试了4次,训练时间从24小时到48小时不等

### distance
使用`Vector_Distance.py`代码进行距离的计算,其中需要在代码中指定训练好的词向量的文件位置


