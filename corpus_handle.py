'''
Created on Thursday July
__author__ : 'jdlimingyang@jd.com'
'''

import numpy as np
import re
import itertools
from collections import Counter

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels():
    '''
    加载类别数据，分词并添加标签
    return: 分词后的句子 和 对定的标签
    '''
    who_examples = list(open('C:\\Users\\jrlimingyang\\PycharmProjects\\question_classifier_cnn\\corpus\\qa_data.who', 'r').readlines())
    who_examples = [s.strip() for s in who_examples]

    when_examples = list(open('C:\\Users\\jrlimingyang\\PycharmProjects\\question_classifier_cnn\\corpus\\qa_data.when', 'r').readlines())
    when_examples = [s.strip() for s in when_examples]

    what_examples = list(open('C:\\Users\\jrlimingyang\\PycharmProjects\\question_classifier_cnn\\corpus\\qa_data.what', 'r').readlines())
    what_examples = [s.strip() for s in what_examples]

    # split by words
    x_text = who_examples + when_examples + what_examples # list 格式
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    who_labels = [[1,0,0] for _ in who_examples]
    when_labels = [[0,1,0] for _ in when_examples]
    what_labels = [[0,0,1] for _ in what_examples]

    y = np.concatenate([who_examples, when_examples, what_examples], 0)
    return [x_text, y]

def pad_sentences(sentences, padding_word='<PAD/>'):
    '''
    将所有句子都pad到相同长度，长度取决于最长的句子
    :return: pad后的sentences
    '''
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + num_padding * [padding_word]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    '''
    建立词到索引和索引到词的映射关系
    :return: word2index 和 index2word 字典
    '''
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping word to index
    vocabulary = {x:i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    '''
    将sentence用词的索引表示（表示成数字）
    '''
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences ])
    y = np.array(labels)
    return [x, y]

def load_data():
    '''
    加载并处理数据集
    :return: 句子的vector, labels, 词和索引的映射和索引到词的映射关系
    '''
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    产生训练的 batch 数据
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # 将每个batch内的数打乱
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index : end_index]
