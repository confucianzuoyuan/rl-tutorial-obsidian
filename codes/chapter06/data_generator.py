import random


class DataGenerator:
    def __init__(self):
        self.vocab = {
            'mark': list('PSE'),
            'number': list('0123456789'),
            'chinese_lower': list('〇一二三四五六七八九'),
            'chinese_upper': list('零壹贰叁肆伍陆柒捌玖'),
            'other': list('数字大写小:=_x'),
        }

        # 解码器
        self.decoder = [j for i in self.vocab.values() for j in i]
        # 编码器
        self.encoder = {j: i for i, j in enumerate(self.decoder)}
        # 标签
        self.label = {
            'number': 0,
            'chinese_lower': 1,
            'chinese_upper': 2,
        }
        # 前缀
        self.prefix = ['数字', '小写', '大写']

    def decode(self, x):
        '''将 `input_ids` 转换成字符串可读数据'''
        return ''.join([self.decoder[i] for i in x])

    def encode(self, text: str):
        '''将字符串转换成对应的 input_ids 列表，假设所有字符都在编码器中'''
        return [self.encoder[ch] for ch in text]

    def get_data(self, prefix: bool):
        '''获取一条数据，prefix为是否带前缀'''
        # 问题和答案对
        question = random.randint(1000, 9999)
        answer = question * 2

        # 将问题和答案转换成字符列表
        question = list(str(question))
        answer = list(str(answer))

        # 随机选择一个标签label
        label = random.choice(list(self.label.keys()))

        # 根据标签类型，将答案换成其它字符集
        answer = [self.vocab[label][int(i)] for i in answer]

        # 将label转换成数字
        label = self.label[label]

        # 组合问题和答案
        if prefix:
            prefix = list(self.prefix[label])
        else:
            prefix = list('__')

        data = prefix + [':'] + question + ['x', '2', '='] + answer
        # 编码成 `input_ids`
        data = [self.encoder[i] for i in data]
        data = [self.encoder['S']] + data + [self.encoder['E']]

        return label, data

    def get_batch_data(self, prefix):
        '''获取一批数据，64条'''
        batch = [self.get_data(prefix) for _ in range(64)]

        batch_labels = [i[0] for i in batch]
        batch_datas = [i[1] for i in batch]

        return batch_labels, *self.batch_pad(batch_datas)

    def batch_pad(self, batch_datas):
        '''对一批数据的每一条添加padding `P`'''
        # 找出一批数据中最长的一条的长度
        max_length = max([len(data) for data in batch_datas])

        input_ids = []
        attention_mask = []
        for data in batch_datas:
            attention_mask.append([1] * len(data) + [0]
                                  * (max_length - len(data)))
            input_ids.append(data + [self.encoder['P']]
                             * (max_length - len(data)))

        return input_ids, attention_mask
