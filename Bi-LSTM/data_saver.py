import csv, re, string
import tensorflow as tf
import gensim
import numpy as np


def _read_csv(input_file):
    """
    read csv file,get data
    :param input_file:
    :return:
    """
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f)
        lines = []
        for line in reader:
            lines.append(line)
        return lines[1:]  # remove header


def sentence_split(sentence, max_length):
    """
    remove punctuation and split sentence.return list of words
    :param sentence:
    :return:
    """
    # sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+'", "", sentence)
    sentence = [x for x in sentence if x not in string.punctuation]
    sentence = ''.join(sentence)
    words = sentence.split()
    if max_length == 0:
        return words
    else:
        if len(words) > max_length:
            words = words[:max_length]
        elif len(words) < max_length:
            words = words + [" "] * (max_length - len(words))
        return words


def embedding_sentence(model_file, input_file, max_length, embedding_dim, save_path):
    """
    get data set and save to tfrecord
    :param data_dir:
    :return:
    """
    lines = _read_csv(input_file)
    split_lines = []
    label_list = []
    for line in lines:
        split_lines.append(sentence_split(line[1], max_length))
        label_list.append(int(line[2]))
    del lines

    writer = tf.python_io.TFRecordWriter(save_path)
    for index, line in enumerate(split_lines):
        bytes_words = []
        for word in line:
            bytes_words.append(str.encode(word))
        # vector = []
        # for word in line:
        #     if word in model:
        #         vector.append(model[word])
        # vector = np.array(vector) / len(line)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":
                tf.train.Feature(int64_list=tf.train.Int64List(value=[label_list[index]])),
            "features":
                tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_words))
        }))
        writer.write(example.SerializeToString())


def build_embedding_model(glove_file, gensim_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        num_lines = 0
        for line in f:
            num_lines += 1
    dims = 300
    gensim_first_line = "{} {}".format(num_lines, dims)
    with open(glove_file, 'r', encoding='utf-8') as fin:
        with open(gensim_file, 'w', encoding='utf-8') as fout:
            fout.write(gensim_first_line + '\n')
            for line in fin:
                fout.write(line)


def build_vocab(model_file, data_file, vocab_path):
    # load glove model
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file)
    lines = _read_csv(data_file)
    vocab = []
    for line in lines:
        vocab.extend(sentence_split(line[1], 0))
    vocab = set(vocab)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for word in vocab:
            if word in model:
                f.write(word + ' ' + ' '.join([str(x) for x in model[word]]) + '\n')


if __name__ == '__main__':
    glove_file = './glove.840B.300d/glove.840B.300d.txt'
    gensim_file = './glove.840B.300d/glove_model.txt'
    dev_input_file = '../train_data/dev.csv'
    embedding_dim = 300
    dev_save_path = '../train_data/dev.tf_record'
    train_input_file = '../train_data/train.csv'
    train_save_path = '../train_data/train.tf_record'
    data_file = '../train_data/deal_train_data.csv'
    vocab_path = '../train_data/vocab.txt'
    # build_embedding_model(glove_file, gensim_file)
    # embedding_sentence(gensim_file, dev_input_file, 30, embedding_dim, dev_save_path)
    # embedding_sentence(gensim_file, train_input_file, 30, embedding_dim, train_save_path)
    build_vocab(gensim_file, data_file, vocab_path)
