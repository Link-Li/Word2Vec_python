"""
TODO
"""
import numpy as np
from vocab.Vocab import Vocab
from multiprocessing import Pool
from multiprocessing import Manager
import multiprocessing
multiprocessing.set_start_method('spawn', True)
import random
import os
import time
import copy


class Word2Vec(object):
    def __init__(self):
        # 用于训练的数据集
        self.train_file_name = 'data/text8'
        # self.train_file_name = 'data/text8_mini'
        # self.train_file_name = 'data/test.txt'
        # 训练好的词向量
        self.output_file_name = 'data/text_vector_test.txt'
        # 将处理好的单词直接读取，省去了进行数据集预处理的步骤
        # self.read_vocab_file_name = 'data/text8_vocab_test_1000_5.txt'
        # self.read_vocab_file_name = 'data/text8_mini_vocab_test.txt'
        # self.read_vocab_file_name = 'data/test_vocab_test.txt'
        self.read_vocab_file_name = ''
        # 将数据集中处理好的单词单独存成文件
        self.save_vocab_file_name = 'data/text8_vocab_test_1000_5.txt'
        # self.save_vocab_file_name = 'data/text8_mini_vocab_test.txt'
        # self.save_vocab_file_name = 'data/test_vocab_test.txt'
        # self.save_vocab_file_name = ''
        # 设置学习率
        self.alpha = Manager().Value('d', 0.025)
        self.start_alpha = 0.025
        # 是否使用cbow，还是使用skip，1表示使用cbow
        self.cbow = 1
        # 词向量的大小，一般使用200维
        self.layer1_size = 10
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
        self.iter_size = 2
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
        self.word_count_actual = Manager().Value('l', 0)
        # 目前已经初始化了多少syn
        self.syn_count_actual = Manager().Value('l', 0)


        # 定义存储单词的变量
        self.vocab_name = []      # 单词的具体内容
        self.vocab_cn = []        # 单词出现的频率
        self.vocab_point = []     # 单词路径中每个节点对应的index值，包括根节点
        self.vocab_code = []      # 单词的哈夫曼编码，也就是单词路径中的左边还是右边，不包括根节点
        self.vocab_code_len = []  # 哈夫曼编码的长度

        # self.syn1 = multiprocessing.Array('d', range(50 * 2))


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
        # self.InitNet()
        #
        # # print('!!!!!!!!!!!!!!!!!!!')
        #
        # self.start_alpha = self.alpha.value
        # now_time = time.time()
        # p = Pool(self.num_thread)
        # for i in range(self.num_thread):
        #     p.apply_async(TrainModelThread, args=(self.num_thread, self.word_count_actual, i,
        #                                           self.syn0, self.syn1, self.vocab_size,
        #                                           self.layer1_size, self.train_file_name, self.alpha,
        #                                           self.start_alpha, self.iter_size, now_time,
        #                                           self.train_word_size, self.file_size, self.vocab_name,
        #                                           self.sample, self.vocab_cn, self.vocab_code,
        #                                           self.vocab_point, self.vocab_code_len, self.max_sentence_length,
        #                                           self.window, self.max_exp, self.expTable,
        #                                           self.exp_table_size))
        # p.close()
        # p.join()

        # TrainModelThread(self.num_thread, self.word_count_actual, 0,
        #                  self.syn0, self.syn1, self.vocab_size,
        #                  self.layer1_size, self.train_file_name, self.alpha,
        #                  self.start_alpha, self.iter_size, now_time,
        #                  self.train_word_size, self.file_size, self.vocab_name,
        #                  self.sample, self.vocab_cn, self.vocab_code,
        #                  self.vocab_point, self.vocab_code_len, self.max_sentence_length,
        #                  self.window, self.max_exp, self.expTable,
        #                  self.exp_table_size)

        # print('!!!!!!!!!!!!!!!!!!!!!')

    def readVocab(self):
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
        with open(self.save_vocab_file_name, 'w', encoding='utf-8') as f_save_vocab_file:
            str_write_into_file = ''
            row_count = 1000
            # 因为在下面的for循环中，当i=0的时候，不会把i=0的数据存进去
            # f_save_vocab_file.write(self.vocab_name[0] + ' ' + str(self.vocab_cn[0]) + '\n' )
            vocab_cn_temp = len(self.vocab_cn)
            for i in range(vocab_cn_temp):
                str_write_into_file += self.vocab_name[i] + ' ' + str(self.vocab_cn[i]) + ' '
                if i % row_count == 0:
                    f_save_vocab_file.write(str_write_into_file + '\n')
                    str_write_into_file = ''
            f_save_vocab_file.write(str_write_into_file + '\n')

    def InitNet(self):
        # 用来存储每个词的词向量
        self.syn0 = Manager().Array('d', range(self.vocab_size * self.layer1_size))
        # 用来存储哈弗曼树中的非叶子节点的词向量，因为哈弗曼树的特性，
        # 所以词向量的个数总是比非叶子节点多一个，这里申请vocab_size个，足够使用了
        self.syn1 = Manager().Array('d', range(self.vocab_size * self.layer1_size))
        # next_random = 1

        """
        # 先注释一下，回来解开，因为这里太计算太慢了
        # 这里是单线程的计算
        for i in range(self.vocab_size):
            for j in range(self.layer1_size):
                next_random = next_random * 25214903917 + 11
                index = i * self.layer1_size + j
                self.syn0[index] = (((next_random & 0xFFFF) / 65536.0) - 0.5) / self.layer1_size
                self.syn1[index] = 0
            if i % 500 == 0:
                print('初始化变量个数：%d' % i)
        """
        # """
        # 先注释一下，回来解开，因为这里太计算有点慢
        # 多线程初始化版本
        p = Pool(self.num_thread + 10)
        for i in range(self.num_thread + 10):
            p.apply_async(InitSynThread, args=(self.num_thread, i, self.syn0, self.syn1,
                                               self.vocab_size, self.layer1_size, self.syn_count_actual))
        p.close()
        p.join()

        # for i in range(self.vocab_size):
        #     for j in range(self.layer1_size):
        #         ii = i * self.layer1_size + j
        #         if self.syn0[ii] != -1:
        #             print('syn0:', self.syn0[ii])
        #         if self.syn1[ii] != -1:
        #             print('syn1:', self.syn1[ii])
        #     if i % 500 == 0:
        #         print('检查变量个数：%d' % i)
        # """

        self.vocab_code = np.array(np.zeros((self.vocab_size, self.max_code_length)), dtype='int64')
        self.vocab_point = np.array(np.zeros((self.vocab_size, self.max_code_length)), dtype='int64')
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

    # def ReadWord(self, f_file):
    #     """
    #     读取文件中的单词，一个字符一个字符的读取完整的单词
    #     这里可以将使用了空格或者制表符的单词一个一个的分离出来
    #     :param f_file: 文件指针
    #     :return: 返回一个单词，或者返回一个空的字符串''，返回空的字符串说明这次没有分离出来单词
    #     """
    #     word = ''
    #     while self.feof(self.file_size, f_file):
    #         ch = f_file.read(1)
    #         # 如果遇到这三个字符就退出读取文件，因为这意味着这个单词读完了或者这个句子读完了，
    #         # 因为一般而言，文件中一个句子结束，会有一个回车，而单词之间一般使用空格或者制表符分离
    #         if ch == ' ' or ch == '\t' or ch == '\n':
    #             break
    #         word += ch
    #     return word

    # def feof(self, file_size, f_file):
    #     """
    #     判断是不是读到了文件尾
    #     :param file_size:  文件大小
    #     :param f_file: 文件指针
    #     :return: 0表示读到了文件尾，1表示没有读到文件尾
    #     """
    #     if file_size == f_file.tell():
    #         return 0
    #     else:
    #         return 1

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

        :return:
        """
        # 主要用来存储每个叶子节点的路径的方向，就是是左还是右
        code_temp = np.array(np.zeros(self.max_code_length), dtype='int64')
        # 记录每个叶子节点的路径，叶子节点都是单词，非叶子节点都不是单词，
        # 但是记录了非叶子节点的位置，也就知道了如何到达叶子节点
        point_temp = np.array(np.zeros(self.max_code_length), dtype='int64')
        # 节点对应的频率
        count_temp = np.array(np.zeros(self.vocab_size * 2 + 1), dtype='int64')
        # 记录每个节点是左节点还是右节点
        binary_temp = np.array(np.zeros(self.vocab_size * 2 + 1), dtype='int64')
        # 记录index的父节点，index是索引
        parent_node_temp = np.array(np.zeros(self.vocab_size * 2 + 1), dtype='int64')

        for i in range(self.vocab_size):
            count_temp[i] = self.vocab_cn[i]
        for i in range(self.vocab_size, self.vocab_size * 2):
            count_temp[i] = 1e15

        pos1 = self.vocab_size - 1
        pos2 = self.vocab_size
        min1i = 0
        min2i = 0

        for i in range(self.vocab_size-1):
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
            b = a
            i = 0

            while True:
                code_temp[i] = binary_temp[b]
                point_temp[i] = b
                i += 1
                b = parent_node_temp[b]
                if b == (self.vocab_size*2 -2):
                    break
            self.vocab_code_len.append(i)
            self.vocab_point[a][0] = self.vocab_size - 2

            for b in range(i):
                self.vocab_code[a][i - b - 1] = code_temp[b]
                self.vocab_point[a][i - b] = point_temp[b] - self.vocab_size

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

def InitSynThread(num_thread, index, syn0, syn1, vocab_size, layer1_size, syn_count_actual):
    print('\rinit syn thread %d start' % index, end='')
    # global syn_count_actual
    every_segment = int(vocab_size / num_thread)
    start_index = int(every_segment * index)
    if num_thread == (index + 1):
        stop_index = vocab_size
    else:
        stop_index = start_index + every_segment
    next_random = index + 1
    for i in range(start_index, stop_index):
        for j in range(layer1_size):
            # next_random = next_random * 25214903917 + 11
            next_random = (next_random * 25214903917 + 11) % 18446744073709551615
            # while next_random > 18446744073709551615:
            #     next_random -= 18446744073709551615
            ii = i * layer1_size + j
            syn0[ii] = (((next_random & 0xFFFF) / 65536.0) - 0.5) / layer1_size
            # syn0[ii] = -1
            syn1[ii] = 0

        if i % 500 == 0:
            syn_count_actual.value += 500
            print('\r初始化变量个数: i=%10d   syn_count_actual=%10d' % (i, syn_count_actual.value), end='')
    print('\rinit syn thread %d stop' % index, end='')


def TrainModelThread(num_thread, word_count_actual, index,
                     syn0, syn1, vocab_size,
                     layer_size, train_file_name, alpha,
                     start_alpha, iter_size, start_time,
                     train_word_size, file_size, vocab_name,
                     sample, vocab_cn, vocab_code,
                     vocab_point, vocab_code_len, max_sentence_length,
                     window, max_exp, expTable,
                     exp_table_size):
    neu1 = np.array(np.zeros(layer_size), dtype='float64')
    neu1e = np.array(np.zeros(layer_size), dtype='float64')
    start_alpha = alpha.value
    sentence_length = 0
    sentence_position = 0
    sentence = np.array(np.zeros(max_sentence_length + 1), dtype='int64')
    next_random = index
    local_iter_size = iter_size

    # 目前已经训练到了第几个word
    word_count = 0
    # 上次打印的时候已经训练到了第几个word
    last_word_count = 0
    with open(train_file_name, encoding='utf-8') as f_train_file:
        f_train_file.seek(int(file_size / num_thread) * index, 0)
        while True:
            if word_count - last_word_count > 1000:
                word_count_actual.value += word_count - last_word_count
                last_word_count = word_count
                print('\rthread num: %d    Alpha: %f    Progress: %.2f%%    Words/thread/sec: %.2fk    '
                      %(index, alpha.value,
                        (word_count_actual.value / (iter_size * train_word_size + 1)) * 100,
                        (word_count_actual.value / ((time.time() - start_time + 1) * 1000 * num_thread))), end="")
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
                    if word_index == 0:
                        break
                    if sample > 0:
                        ran = (np.sqrt(vocab_cn[word_index] / (sample * train_word_size)) + 1) * (sample * train_word_size) / vocab_cn[word_index]
                        # next_random = next_random * 25214903917 + 11
                        next_random = (next_random * 25214903917 + 11) % 18446744073709551615
                        if ran < (next_random & 0xFFFF) / 65536.0:
                            continue
                    sentence[sentence_length] = word_index
                    sentence_length += 1
                    if sentence_length >= max_sentence_length:
                        break
                sentence_position = 0
            if feof(file_size, f_train_file) == 0 or word_count > train_word_size / num_thread:
                word_count_actual.value += word_count - last_word_count
                local_iter_size -= 1
                if local_iter_size == 0:
                    # print('结束啦 %d    iter: %d' % (index, local_iter_size))
                    break
                word_count = 0
                last_word_count = 0
                sentence_length = 0
                f_train_file.seek(int(file_size / num_thread) * index, 0)
                continue
            word_index = sentence[sentence_position]
            if word_index == -1:
                continue
            for i in range(layer_size):
                neu1[i] = 0
                neu1e[i] = 0
            # next_random = next_random * 25214903917 + 11
            next_random = (next_random * 25214903917 + 11) % 18446744073709551615
            b = next_random % window
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
                    move_position = last_word * layer_size
                    for j in range(layer_size):
                        neu1[j] += syn0[int(j + move_position)]
                    cw += 1
            
            if cw > 0:
                for i in range(layer_size):
                    neu1[i] /= cw
                for i in range(vocab_code_len[word_index]):
                    f = 0
                    l2 = vocab_point[word_index][i] * layer_size
                    for j in range(layer_size):
                        f += neu1[j] * syn1[j + l2]
                    if f <= -max_exp or f >= max_exp:
                        continue
                    else:
                        f = expTable[int((f + max_exp) * (exp_table_size / max_exp / 2))]
                    g = (1 - vocab_code[word_index][i] -f) * alpha.value
                    for j in range(layer_size):
                        neu1e[j] += g * syn1[j + l2]
                    for j in range(layer_size):
                        syn1[j + l2] += g * neu1[j]
                for i in range(b, window * 2 + 1 - b):
                    if i != window:
                        c = sentence_position - window + i
                        if c < 0 or c >= sentence_length:
                            continue
                        last_word = sentence[c]
                        if last_word == -1:
                            continue
                        move_position = last_word * layer_size
                        for j in range(layer_size):
                            syn0[j + move_position] += neu1e[j]


            # while next_random > 18446744073709551615:
            #     next_random -= 18446744073709551615
            # a = 1
            # for p in range(0):
            #     a += (2 ** p + p) / (3 ** p + p)
            # word_count += 1
            sentence_position += 1
            if sentence_position >= sentence_length:
                sentence_length = 0
                continue





def task(num_thread, index, syn1_thread, vocab_size, layer_size):
    print('进程%d' % index)
    a = 50 / num_thread
    for i in range(vocab_size):
        for j in range(layer_size):
            for p in range(0):
                a += (2 ** p + p) / (3 ** p + p)
            syn1_thread[random.randint(0, vocab_size) * 2 + j] = index
    print("进程：%d结束" % index)



if __name__ == '__main__':
    start_time = time.time()
    word2Vec = Word2Vec()
    # 初始化生产sigmoid的值
    word2Vec.init_expTable()
    word2Vec.Main()
    print('用时：%d' % (time.time() - start_time))
