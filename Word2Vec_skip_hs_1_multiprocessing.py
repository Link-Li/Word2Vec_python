"""
Name: Word2Vec_skip_hs_1_multiprocessing.py
Date: 19-8-17 下午1:34
TODO:
"""

import numpy as np
import multiprocessing as mp
mp.set_start_method('spawn', True)
import os
import time
import copy


class Word2Vec(object):
    def __init__(self):
        # 用于训练的数据集
        self.train_file_name = 'data/text8'
        # self.train_file_name = 'data/text8_mini'
        # self.train_file_name = 'data/text8_mini_2'
        # self.train_file_name = 'data/test.txt'
        # 训练好的词向量
        self.output_file_name = 'data/text8_vector_test_1.npz'
        # self.output_file_name = 'data/text8_mini_vector_multi_1.npz'
        # self.output_file_name = 'data/text8_mini_2_vector_test.npz'
        # self.output_file_name = 'data/text_vector_test.npz'
        # 将处理好的单词直接读取，省去了进行数据集预处理的步骤
        self.read_vocab_file_name = 'data/text8_vocab_test_1000_5.txt'
        # self.read_vocab_file_name = 'data/text8_mini_vocab_test.txt'
        # self.read_vocab_file_name = 'data/text8_mini_2_vocab_test.txt'
        # self.read_vocab_file_name = 'data/test_vocab_test.txt'
        # self.read_vocab_file_name = ''
        # 将数据集中处理好的单词单独存成文件
        # self.save_vocab_file_name = 'data/text8_vocab_test_1000_5.txt'
        # self.save_vocab_file_name = 'data/text8_mini_vocab_test.txt'
        # self.save_vocab_file_name = 'data/text8_mini_2_vocab_test.txt'
        # self.save_vocab_file_name = 'data/test_vocab_test.txt'
        self.save_vocab_file_name = ''
        # 设置学习率
        self.alpha = mp.Value('d', 0.025)
        # self.alpha = 0.025
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
        self.num_thread = 10
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
        self.word_count_actual = mp.Value('l', 0)
        # self.word_count_actual = 0
        # 目前已经初始化了多少syn
        # self.syn_count_actual = Manager().Value('l', 0)


        # 定义存储单词的变量
        self.vocab_name = []      # 单词的具体内容
        self.vocab_cn = []        # 单词出现的频率
        self.vocab_point = []     # 单词路径中每个节点对应的index值，包括根节点
        self.vocab_code = []      # 单词的哈夫曼编码，也就是单词路径中的左边还是右边，不包括根节点
        self.vocab_code_len = []  # 哈夫曼编码的长度
        self.syn0 = []
        self.syn1 = []
        self.shape_syn = []


    def Main(self):
        """
        主要进行整体代码的逻辑控制
        :return:
        """
        # 如果有处理好的文件，那么就直接读取处理好的文件，直接读取的文件是排序好的，不需要再次排序的
        if self.read_vocab_file_name.strip():
            # 获取文件的大小，用于判断是不是读到了文件尾，以及多线程运行的分割标准等
            self.file_size = os.path.getsize(self.train_file_name)
            self.readVocab()
        # 如果没有处理好的文件，那么就先处理，获取了词汇的数据，然后再进行下面的步骤
        else:
            self.LearnVocabFromTrainFIle()
        if self.debug_mode > 0:
            print('vocab size : %d' % self.vocab_size)
            print('word in file : %d\n' % self.train_word_size)
        # 如果有可以存储文件的地址，那么就把分析好的词汇表存起来，这里只包含了单词和单词的频率
        if self.save_vocab_file_name.strip():
            self.SaveVocab()
        # 这个是保存训练好的单词的词向量
        # 目前的构想是在文件第一行存储一些词向量的全局信息，然后之后每行开头是单词，之后是词向量，中间使用空格分割
        if self.output_file_name.strip() == '':
            print('please input output_file!')
            return
        # 初始化网络，特别是创建哈弗曼树和一些哈弗曼树需要使用的变量
        self.InitNet()

        # self.start_alpha = self.alpha
        now_time = time.time()
        # p = Pool(self.num_thread)
        p = mp.Process()
        for i in range(self.num_thread):
            p = mp.Process(target=TrainModelThread, args=(self.num_thread, self.word_count_actual, i,
                                                          self.syn0, self.syn1, self.vocab_size,
                                                          self.layer1_size, self.train_file_name, self.alpha,
                                                          self.shape_syn, self.iter_size, now_time,
                                                          self.train_word_size, self.file_size, self.vocab_name,
                                                          self.sample, self.vocab_cn, self.vocab_code,
                                                          self.vocab_point, self.vocab_code_len,
                                                          self.max_sentence_length,
                                                          self.window, self.max_exp, self.expTable,
                                                          self.exp_table_size))
            p.start()

        p.join()

        self.SaveVocabVector()

    def readVocab(self):
        """
        读取存储好的单词的文件,但是如果存单词的时候考虑使用np.savez(),这里的代码估计会更简单
        :return:
        """
        self.vocab_name.clear()
        self.vocab_cn.clear()
        with open(self.read_vocab_file_name, encoding='utf-8') as f_read_vocab_file:
            vocab_lines = f_read_vocab_file.readlines()
            for vocab_line in vocab_lines:
                vocab_array = vocab_line.strip().split(' ')
                for index in range(0, len(vocab_array), 2):
                    self.vocab_name.append(vocab_array[index])
                    self.vocab_cn.append(int(vocab_array[index + 1]))
        self.vocab_size = len(self.vocab_cn)
        self.train_word_size = 0
        for i in self.vocab_cn:
            self.train_word_size += i

    def LearnVocabFromTrainFIle(self):
        """
        这个函数将会从训练数据中分离出需要的词
        :return:
        """
        with open(self.train_file_name, 'r', encoding='utf-8') as f_train_file:
            # 首先把回车符放到单词列表中，虽然我现在也不知道有啥用
            self.AddWordToVocab('</s>')
            self.train_word_size += 1
            # 获取文件的大小，用于判断是不是读到了文件尾，以及多线程运行的分割标准等
            self.file_size = os.path.getsize(self.train_file_name)
            while feof(self.file_size, f_train_file):
                word_temp = ReadWord(f_train_file, self.file_size)
                # 当没有读到单词的时候就跳过
                if word_temp == '':
                    continue
                self.AddWordToVocab(word_temp)
                self.train_word_size += 1
                if self.vocab_size > self.vocab_hash_size:
                    # 原来的代码，这里是要对小于min_reduce的词都删除，但是考虑到目前还用不到这个
                    # 所以先不写了
                    pass
                if self.debug_mode > 0 and (self.train_word_size % 100000 == 0):
                    print('find %dK vocab' % (self.train_word_size / 1000))
            # 根据词汇的频率对单词进行排序,必须排序后才能将词汇表存入文件
            self.SortVocab()

    def SaveVocab(self):
        """
        建议之后考虑使用np.savez()进行存储
        :return:
        """
        with open(self.save_vocab_file_name, 'w', encoding='utf-8') as f_save_vocab_file:
            str_write_into_file = ''
            row_count = 1000
            vocab_cn_temp = len(self.vocab_cn)
            for i in range(vocab_cn_temp):
                str_write_into_file += self.vocab_name[i] + ' ' + str(self.vocab_cn[i]) + ' '
                if i % row_count == 0:
                    f_save_vocab_file.write(str_write_into_file + '\n')
                    str_write_into_file = ''
            # 最后一次循环，可能数据的数量不能够整除，余出来的数据就再存一次
            # 不过这里如果使用np.savez()估计会更简单一点
            f_save_vocab_file.write(str_write_into_file + '\n')

    def InitNet(self):
        """
        初始化syn0和syn1，以及一些用来存储哈弗曼树中信息的变量
        :return:
        """
        # 用来存储每个词的词向量
        self.syn0 = np.array(np.zeros(self.vocab_size * self.layer1_size))
        # 用来存储哈弗曼树中的非叶子节点的词向量，因为哈弗曼树的特性，
        # 所以词向量的个数总是比非叶子节点多一个，这里申请vocab_size个，足够使用了
        self.syn1 = np.array(np.zeros(self.vocab_size * self.layer1_size))

        # self.InitSyn()

        print('init syn start')
        rand_max = 0.5 / self.layer1_size
        self.syn0 = np.expand_dims(np.random.uniform(-rand_max, rand_max, size=[self.vocab_size * self.layer1_size]), axis=1)
        self.syn1 = np.expand_dims(np.zeros(self.vocab_size * self.layer1_size), axis=1)
        self.shape_syn = self.syn0.shape
        self.syn0.shape = self.syn0.size
        self.syn1.shape = self.syn1.size
        self.syn0 = mp.Array('d', self.syn0)
        self.syn1 = mp.Array('d', self.syn1)
        print('init syn stop')

        self.vocab_code = np.array(np.zeros((self.vocab_size, self.max_code_length)), dtype='int64')
        self.vocab_point = np.array(np.zeros((self.vocab_size, self.max_code_length)), dtype='int64')
        # 构建哈弗曼树
        self.CreateBinaryTree()

    def AddWordToVocab(self, word):
        """
        将单词添加到单词表中，并实时更新单词实际数量
        :param word:  要添加的单词
        :return:
        """
        vocab_index = 0
        # 先判断单词存不存在单词表中
        try:
            vocab_index = self.vocab_name.index(word)
        except ValueError:
            vocab_index = -1
        if vocab_index == -1:
            self.vocab_name.append(word)
            self.vocab_cn.append(1)
            self.vocab_size += 1
            # self.vocab_point.append(list(range(self.vocab_size, self.vocab_size+1)))
            # self.vocab_code.append(list(range(self.vocab_size, self.vocab_size+1)))
            # self.vocab_code_len.append(1)
        else:
            self.vocab_cn[vocab_index] += 1

    def SortVocab(self):
        """
        根据单词的词频进行排序，小于self.min_count的单词都会被舍弃掉
        :return:
        """
        # 因为这里先将原来的词汇表清除掉，所以使用深拷贝，因为有些变量可能涉及到二级列表
        vocab_name_copy = copy.deepcopy(self.vocab_name)
        vocab_cn_copy = copy.deepcopy(self.vocab_cn)
        # 返回的其实是一个从小到大的排序索引，但是我们需要的是从大到小的
        cn_sort_index = np.argsort(np.array(self.vocab_cn[1:]), )+1
        cn_sort_index = np.append(cn_sort_index, 0)
        i = 0
        # 清除原来词汇表的信息，然后重新填入
        self.vocab_name.clear()
        self.vocab_cn.clear()
        # 判断哈夫曼编码是否需要排序
        if self.vocab_code:  # 哈夫曼编码需要排序
            vocab_point_copy = copy.deepcopy(self.vocab_point)
            vocab_code_copy = copy.deepcopy(self.vocab_code)
            vocab_code_len_copy = copy.deepcopy(self.vocab_code_len)
            self.vocab_point.clear()
            self.vocab_code.clear()
            self.vocab_code_len.clear()
            # 因为需要从大到小，所以这里就反向输出cn_sort_index
            for index in cn_sort_index[::-1]:
                if vocab_cn_copy[index] < self.min_count and index != 0:
                    break
                self.vocab_name.append(vocab_name_copy[index])
                self.vocab_cn.append(vocab_cn_copy[index])
                self.vocab_point.append(vocab_point_copy[index])
                self.vocab_code.append(vocab_code_copy[index])
                self.vocab_code_len.append(vocab_code_len_copy[index])
                i += 1
        else:
            # 因为需要从大到小，所以这里就反向输出cn_sort_index
            for index in cn_sort_index[::-1]:
                if vocab_cn_copy[index] < self.min_count and index != 0:
                    break
                self.vocab_name.append(vocab_name_copy[index])
                self.vocab_cn.append(vocab_cn_copy[index])
                i += 1
        self.vocab_size = len(self.vocab_cn)
        self.train_word_size = 0
        for i in self.vocab_cn:
            self.train_word_size += int(i)

    def CreateBinaryTree(self):
        """
        创建哈弗曼树,原作者使用的方法非常巧妙,构建的时候比较抽象,并没有使用指针之类的,
        而是直接使用的数组去存的,一次性构建好了之后,将单词在哈弗曼树中的位置等信息存储好,
        然后就将哈弗曼树的构建过程给扔了,之后使用syn0和syn1的下标值来代替这里的哈弗曼树
        :return:
        """
        # 主要用来存储每个叶子节点的路径的方向，就是是左还是右
        code_temp = np.array(np.zeros(self.max_code_length), dtype='int64')
        # 记录每个叶子节点的路径，叶子节点都是单词，非叶子节点都不是单词，
        # 但是记录了非叶子节点的位置，也就知道了如何到达叶子节点
        point_temp = np.array(np.zeros(self.max_code_length), dtype='int64')
        # 节点对应的频率,一分为2,将前面的作为存单词的频率,后面的一半是节点所代表的频率
        count_temp = np.array(np.zeros(self.vocab_size * 2 + 1), dtype='int64')
        # 记录每个节点是左节点还是右节点,和上面一样,前半部分存单词(叶子节点)的位置,后半部分存节点(非叶子节点)的位置
        binary_temp = np.array(np.zeros(self.vocab_size * 2 + 1), dtype='int64')
        # 记录index的父节点，index是索引
        parent_node_temp = np.array(np.zeros(self.vocab_size * 2 + 1), dtype='int64')

        for i in range(self.vocab_size):
            count_temp[i] = self.vocab_cn[i]
        for i in range(self.vocab_size, self.vocab_size * 2):
            count_temp[i] = 1e15

        pos1 = self.vocab_size - 1  # 从单词最后开始向前遍历
        pos2 = self.vocab_size  # 从单词结尾开始,也就是非叶子节点开始向前遍历
        min1i = 0
        min2i = 0

        for i in range(self.vocab_size-1):
            # 两个if-else,因为一次要选出来两个节点,然后构建一个哈弗曼树的子树(3个节点)
            # 这里构建的时候不是连续的,相当于构建的时候,能构建一个子树就构建一个子树(3个节点)
            # 当两个子树能连接成一个稍微大点的子树(7个节点)的时候,再连接起来,然后循环下去,构建越来越大的子树
            # 反正最后会构建成一个完整的哈弗曼树
            if pos1 >= 0:
                if count_temp[pos1] < count_temp[pos2]:
                    min1i = pos1
                    pos1 -= 1
                else:
                    min1i = pos2
                    pos2 += 1
            else:
                min1i = pos2
                pos2 += 1
            if pos1 >= 0:
                if count_temp[pos1] < count_temp[pos2]:
                    min2i = pos1
                    pos1 -= 1
                else:
                    min2i = pos2
                    pos2 += 1
            else:
                min2i = pos2
                pos2 += 1

            count_temp[self.vocab_size + i] = count_temp[min1i] + count_temp[min2i]
            parent_node_temp[min1i] = self.vocab_size + i
            parent_node_temp[min2i] = self.vocab_size + i
            binary_temp[min2i] = 1

        for a in range(self.vocab_size):
            # 父节点的位置,因为哈弗曼树的特殊性质,父节点肯定不是单词,单词都是叶子节点
            b = a
            i = 0
            while True:
                code_temp[i] = binary_temp[b]
                point_temp[i] = b
                i += 1
                b = parent_node_temp[b]  # 寻找下一个父节点
                if b == (self.vocab_size*2 -2):
                    break
            self.vocab_code_len.append(i)
            self.vocab_point[a][0] = self.vocab_size - 2

            for b in range(i):
                self.vocab_code[a][i - b - 1] = code_temp[b]  #相当于哈夫曼编码
                self.vocab_point[a][i - b] = point_temp[b] - self.vocab_size

    def SaveVocabVector(self):
        self.syn0 = np.frombuffer(self.syn0.get_obj(), dtype=np.float)
        self.syn0.shape = self.shape_syn
        np.savez(self.output_file_name, vocab_name=self.vocab_name, vocab_vector=self.syn0,
                 vocab_size=self.vocab_size, layer_size=self.layer1_size)

    def init_expTable(self):
        for i in range(self.exp_table_size):
            temp = np.exp(((i / float(self.exp_table_size)) * 2 -1) * self.max_exp)
            self.expTable.append(temp / (temp+1))

def feof(file_size, f_file):
        """
        判断是不是读到了文件尾
        :param file_size:  文件大小
        :param f_file: 文件指针
        :return: 0表示读到了文件尾，1表示没有读到文件尾
        """
        if file_size == f_file.tell():
            return 0
        else:
            return 1

def TrainModelThread(num_thread, word_count_actual, index,
                     syn0, syn1, vocab_size,
                     layer1_size, train_file_name, alpha,
                     shape_syn, iter_size, now_time,
                     train_word_size, file_size, vocab_name,
                     sample, vocab_cn, vocab_code,
                     vocab_point, vocab_code_len, max_sentence_length,
                     window, max_exp, expTable,
                     exp_table_size):
    neu1 = np.expand_dims(np.array(np.zeros(layer1_size), dtype='float64'), axis=1)
    neu1e = np.expand_dims(np.array(np.zeros(layer1_size), dtype='float64'), axis=1)
    start_alpha = alpha.value
    sentence_length = 0
    sentence_position = 0
    sentence = np.array(np.zeros(max_sentence_length + 1), dtype='int64')
    local_iter_size = iter_size
    syn0 = np.frombuffer(syn0.get_obj(), dtype=np.float)
    syn0.shape = shape_syn
    syn1 = np.frombuffer(syn1.get_obj(), dtype=np.float)
    syn1.shape = shape_syn

    # 目前已经训练到了第几个word
    word_count = 0
    # 上次打印的时候已经训练到了第几个word
    last_word_count = 0
    with open(train_file_name, encoding='utf-8') as f_train_file:
        # f_train_file.seek(0, 0)
        f_train_file.seek(int(file_size / num_thread) * index, 0)
        while True:
            if word_count - last_word_count > 10000:
                word_count_actual.value += word_count - last_word_count
                last_word_count = word_count
                # 这里是对学习率的动态更新,逐渐减小学习率
                print('\rAlpha: %f    Progress: %.2f%%    Words/thread/sec: %.2fk    '
                      % (alpha.value,
                         (word_count_actual.value / (iter_size * train_word_size + 1)) * 100,
                         (word_count_actual.value / ((time.time() - now_time + 1) * 1000 * num_thread))), end="")
                # 这里稍微修改一下吧,主要是word_count_actual没法计算了,或者减小train_word_size呢?感觉还是减小一下train_word_size吧
                alpha.value = start_alpha * (1 - word_count_actual.value / (iter_size * train_word_size + 1))
                if alpha.value < start_alpha * 0.0001:
                    alpha.value = start_alpha * 0.0001

            if sentence_length == 0:
                while True:
                    word_index = ReadWordIndex(f_train_file, file_size, vocab_name)
                    if feof(file_size, f_train_file) == 0:
                        break
                    if word_index == -1:
                        continue
                    word_count += 1
                    # 如果是回车,说明这个句子读完了,那么可以跳出循环了
                    if word_index == 0:
                        break
                    # 下采样随机丢弃频繁的单词，同时保持单词的排名不变，随机的跳过一些单词的训练
                    if sample > 0:
                        ran = (np.sqrt(vocab_cn[word_index] / (sample * train_word_size)) + 1) * \
                              (sample * train_word_size) / vocab_cn[word_index]
                        if ran < np.random.rand():
                            continue
                    sentence[sentence_length] = word_index
                    sentence_length += 1
                    if sentence_length >= max_sentence_length:
                        break
                sentence_position = 0
            # 进行变量的重置,为了进行下一次迭代
            if feof(file_size, f_train_file) == 0 or word_count > train_word_size / num_thread:
                word_count_actual.value += word_count - last_word_count
                local_iter_size -= 1
                if local_iter_size == 0:
                    break
                word_count = 0
                last_word_count = 0
                sentence_length = 0
                # f_train_file.seek(0, 0)
                f_train_file.seek(int(file_size / num_thread) * index, 0)
                continue
            word_index = sentence[sentence_position]
            if word_index == -1:
                continue
            neu1 = np.expand_dims(np.array(np.zeros(layer1_size), dtype='float64'), axis=1)
            neu1e = np.expand_dims(np.array(np.zeros(layer1_size), dtype='float64'), axis=1)
            # 用随机数在区间[1，Windows]上来生成一个窗口的大小
            b = np.random.randint(0, window) % window
            cw = 0
            for i in range(b, window * 2 + 1 - b):
                if i != window:
                    c = sentence_position - window + i
                    if c < 0 or c >= sentence_length:
                        continue
                    # if c >= sentence_length:
                    #     continue
                    last_word = sentence[c]
                    if last_word == -1:
                        continue
                    # 投影层,将多个单词投影到neul中
                    move_position = last_word * layer1_size
                    neu1 += syn0[move_position: layer1_size + move_position]
                    cw += 1

            if cw > 0:
                neu1 /= cw
                for i in range(vocab_code_len[word_index]):
                    f = 0
                    l2 = vocab_point[word_index][i] * layer1_size
                    f += np.sum(neu1 * syn1[l2:l2 + layer1_size])
                    if f <= -max_exp or f >= max_exp:
                        continue
                    else:
                        f = expTable[int((f + max_exp) * (exp_table_size / max_exp / 2))]
                    g = (1 - vocab_code[word_index][i] - f) * alpha.value
                    neu1e += g * syn1[l2:l2 + layer1_size]
                    syn1[l2:l2 + layer1_size] += g * neu1
                for i in range(b, window * 2 + 1 - b):
                    if i != window:
                        c = sentence_position - window + i
                        if c < 0 or c >= sentence_length:
                            continue
                        last_word = sentence[c]
                        if last_word == -1:
                            continue
                        move_position = last_word * layer1_size
                        syn0[move_position:move_position + layer1_size] += neu1e

            sentence_position += 1
            if sentence_position >= sentence_length:
                sentence_length = 0
                continue

def ReadWord(f_file, file_size):
    """
    读取文件中的单词，一个字符一个字符的读取完整的单词
    这里可以将使用了空格或者制表符的单词一个一个的分离出来
    :param f_file: 文件指针
    :param file_size: 文件大小
    :return: 返回一个单词，或者返回一个空的字符串''，返回空的字符串说明这次没有分离出来单词
    """
    word = ''
    while feof(file_size, f_file):
        ch = f_file.read(1)
        # 如果遇到这三个字符就退出读取文件，因为这意味着这个单词读完了或者这个句子读完了，
        # 因为一般而言，文件中一个句子结束，会有一个回车，而单词之间一般使用空格或者制表符分离
        if ch == ' ' or ch == '\t' or ch == '\n':
            break
        word += ch
    return word

def ReadWordIndex(f_read_word, file_size, vocab_name):
    vocab_name_temp = ReadWord(f_read_word, file_size)
    vocab_index = -1
    try:
        vocab_index = vocab_name.index(vocab_name_temp)
    except ValueError:
        vocab_index = -1
    return vocab_index


if __name__ == '__main__':
    start_time = time.time()
    word2Vec = Word2Vec()
    # 初始化生产sigmoid的值
    word2Vec.init_expTable()
    word2Vec.Main()
    print()
    print('用时：%d' % (time.time() - start_time))
