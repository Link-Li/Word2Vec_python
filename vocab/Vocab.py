"""
TODO
"""

class Vocab(object):

    def __init__(self):
        # 单词的具体内容
        self.word = ''
        # 单词出现的频率
        self.count = 0
        # 单词路径中每个节点对应的index值，包括根节点
        self.point = []
        # 单词的哈夫曼编码，也就是单词路径中的左边还是右边，不包括根节点
        self.code = []
        # 哈夫曼编码的长度
        self.code_len = 0