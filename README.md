# Word2Vec_python

**中文版的README请参考<a href="https://github.com/Link-Li/Word2Vec_python/blob/master/README-chinese.md" target="_blank">README-chinese</a>**

After a period of study, I started to write these python code, I am still updating it. When I write the code, I mainly refer to the following links

**<a href="https://github.com/dav/word2vec" target="_blank">dav/word2vec</a>**

**<a href="https://github.com/liuwei1206/word2vec" target="_blank">liuwei1206/word2vec</a>**

**<a href="https://github.com/linshouyi/Word2VEC_java" target="_blank">linshouyi/Word2VEC_java</a>**

At the same time, I am especially grateful to the peghoty@163.com who wrtie the **the math in word2vec**, I also put this pdf into the `/pdf` folder. You can learn from it.

### cbow
At present, I finish the cbow's single-thread code. If you need to train it, you can run the code of `Word2Vec_cbow_hs_1_one_thread.py`. You can change the parameter from the code
```
        # 用于训练的数据集
        self.train_file_name = 'data/text8'
        # 训练好的词向量
        self.output_file_name = 'data/text8_vector_test_1.npz'
        # 将处理好的单词直接读取，省去了进行数据集预处理的步骤
        self.read_vocab_file_name = 'data/text8_vocab_test_1000_5.txt'
        # 将数据集中处理好的单词单独存成文件
        self.save_vocab_file_name = ''
        # 设置学习率
        self.alpha = 0.025
        # self.start_alpha = 0.025
        # 是否使用cbow，还是使用skip，1表示使用cbow
        self.cbow = 1
        # 词向量的大小，一般使用200维
        self.layer1_size = 200
        # 窗口大小
        self.window = 8
        # 负采样大小，0表示不使用负采样
        self.negative = 0
        # 是否使用softmax，0表示不使用
        self.hs = 1
        # 下采样率
        self.sample = 1e-4
        # 训练使用的线程数量
        # self.num_thread = 10
        # 训练的迭代次数
        self.iter_size = 15
        # 是否打印输出信息，大于0就打印
        self.debug_mode = 1
        # 单词的最小频率，小于这个频率的单词会被舍弃掉
        self.min_count = 5
        # 用于存放计算的sigmoid的值
        self.expTable = []
        # 用于生成sigmoid分割细度的大小
        self.exp_table_size = 1000
        self.max_string = 100
        # 最大的sigmoid的输入参数[-6, 6]
        self.max_exp = 6
        # 训练过程中最长的句子长度
        self.max_sentence_length = 1000
        # 构建的最深的哈弗曼树
        self.max_code_length = 100
        # 用于训练的单词的个数
        self.vocab_size = 0
        # 用于训练的单词的实际个数，因为一个单词可能出现多次，这里记录了查找单词的操作次数
        self.train_word_size = 0
        # 文件的大小，因为文件大小可以作为多进程划分的依据
        self.file_size = 0
        # 最大的可以分析的词的数量，这里并没有使用哈希存储，这个变量就是表示一下最大的词汇量
        self.vocab_hash_size = 30000000
        # 目前实际已经训练多少word
        self.word_count_actual = 0
```

My cpu is I7-6700HQ, if you first run the code `Word2Vec_cbow_hs_1_one_thread.py`. You will need to set the parameter `save_vocab_file_name`, this parameter will tell the code where to store the processed words. If you use the dataset `text8`, you will take about 40 minutes to find the words from the `text8`, and then store them in the file. If you run the code again, you can set the parameter `read_vocab_file_name`, then it can read the code very quickly.

Then, it will begin train the cbow, it take about 24-48 hours.

Now, I finish the multiprocessing code. You can run the code `Word2Vec_cbow_hs_1_multithread.py`, and use the parameter `num_thread` to set the num of threads. I test the code on i7-9700K, and the speed is about `3.5k/thread/s`, it taking about 2 hours. I also test it on i7-6700HQ, it taking about 6 hours.

Here, I have to make complaints about the python's multiprocessing data sharing problem, the speed is very low. When I using single-processing operation, the speed is about `2.5k/thread/s` on the i7-6700HQ, but when using multiprocessing, the speed is directly reduced by half!

You can find the vocab which is processed in <a href="https://pan.baidu.com/s/1ruOs7RFy140L8L9UHvBKIw" traget="_blank">百度网盘</a>, The password is: fs5t. Where `text8` is the dataset, and the `text8_vocab_test_1000_5.txt` is the word I have processed and it can be read directly.

### skip-hs
You can run the code `Word2Vec_skip_hs_1_multiprocessing.py`, but it takes me about 8 hours. I train it on the i7-9700k and use 8 processing. However, the speed of every process is about `0.8k/s`. 

### cbow-ns
You can run the code `Word2Vec_cbow_ns_1_multiprocessing.py`. This is the negative sampling implementation of cbow. It takes me about 80 minutes on i7-9700k

### distance
You can use the `Vector_Distance.py` code to calculate the distance of words, you need to set the parameter `input_vector_file_name`, which can tell the code where can find the word vector which is trained

You can find the vocab vector which is I trained in <a href="https://pan.baidu.com/s/1ruOs7RFy140L8L9UHvBKIw" traget="_blank">百度网盘</a>, the password is: fs5t. Where the vocab vector is stored in `text8_vector_cbow_1.npz`.



