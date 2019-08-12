"""
TODO 计算和所给单词距离最近的词，这里原来的作者使用的是余弦变量(原作者写成了余弦距离，应该是笔误了)，
     这里计算的方式比价有意思，每次都是先对向量进行处理，
     计算的公式是：cos(A, B) = [A/sqrt(A^2)]*[B/sqrt(B^2)]
"""

import numpy as np


class VectorDistance(object):

    def __init__(self):
        self.input_vector_file_name = 'data/text8_vector_test_1.npz'
        # self.input_vector_file_name = 'data/text8_mini_vector_test_1.npz'
        self.max_size = 2000
        # 展现最接近的词的个数
        self.show_word_count = 40
        # 存储所有词的名字
        self.vocab_name = []
        # 单词数量
        self.vocab_size = 0
        # 存储所有词向量
        self.vocab_vector = []
        # 每个词向量的维度
        self.layer_size = 0
        # # 存放和所给单词最好的距离，一共存10个
        # self.best_distance = np.zeros((self.show_word_count, 1)) - 1
        # # 存放最好距离的单词名字
        # self.best_word = []
        # 存放所输入的单词的向量，或者是句子的向量（句子中每个词的向量加起来的）
        self.input_vector = np.zeros((self.max_size, 1))
        # 存放所输入单词的id，即在self.vocab_name中的位置
        self.input_vocab_id = np.zeros(1000, dtype='int64') - 1
        # 键盘输入的单词或者句子
        self.input_vocab = ''

    def Main(self):
        self.readVocabVector()

        while True:
            self.input_vocab = input('Enter word or sentence (EXIT to break):').split(' ')
            if self.input_vocab[0] == 'EXIT':
                print('goodbye')
                exit(0)
            self.input_vector = np.zeros((self.max_size, 1))

            for index, vocab in enumerate(self.input_vocab):
                word_index_temp = -1
                try:
                    word_index_temp = self.vocab_name.index(vocab)
                    self.input_vocab_id[index] = word_index_temp
                    self.input_vector[0: self.layer_size] += self.vocab_vector[word_index_temp * self.layer_size: word_index_temp * self.layer_size + self.layer_size]
                except ValueError as e:
                    print(e)
                    word_index_temp = -1
                    print('no this vocab!')
                    break
            if word_index_temp == -1:
                continue
            self.input_vector /= np.sqrt(np.sum(self.input_vector ** 2))

            # self.best_distance = np.zeros((self.show_word_count, 1)) - 1
            # self.best_word.clear()

            # 存储每次计算出来的余弦变量的值
            dist = []
            # for input_word_index in self.input_vocab_id:
            for index, vocab in enumerate(self.vocab_name):
                if vocab in self.input_vocab:
                    # 遇到了相同的词，但是为了保证下面排序的时候，顺序是正确的，那么这里用一个负数进行填充
                    dist.append(-2.0)
                    continue
                dist.append(float(np.sum(self.input_vector[0: self.layer_size] *
                                         self.vocab_vector[index * self.layer_size: index * self.layer_size + self.layer_size])))
            # 这里获得的是一个从小到大排列的dist的下标值，下面反向输出就可以得到余弦变量值最大的那几个单词了
            dist_sort_index = np.argsort(dist)
            count_num = 0
            for index in dist_sort_index[::-1]:
                print('vocab: %20s    Cosine distance: %f' % (self.vocab_name[index], dist[index]))
                count_num += 1
                if count_num >= self.show_word_count:
                    break

    def readVocabVector(self):
        data = np.load(self.input_vector_file_name)
        self.vocab_name = data['vocab_name'].tolist()
        self.vocab_vector = data['vocab_vector']
        self.layer_size = data['layer_size']
        self.vocab_size = data['vocab_size']
        for i in range(self.vocab_size):
            index = i * self.layer_size
            self.vocab_vector[index: index + self.layer_size] /= np.sqrt(np.sum(self.vocab_vector[index: index + self.layer_size] ** 2))
        # a = 1
        # len = np.sqrt(np.sum(self.vocab_vector ** 2))
        # self.vocab_vector /= len




if __name__ == '__main__':
    vector_distance = VectorDistance()
    vector_distance.Main()

