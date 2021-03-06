import csv, string
import tensorflow as tf
import gensim
import numpy as np
from data_helper import loadGloVe


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


def embedding_sentence(input_file, save_path, max_length):
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
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":
                tf.train.Feature(int64_list=tf.train.Int64List(value=[label_list[index]])),
            "features":
                tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_words))
        }))
        writer.write(example.SerializeToString())


def embedding_sentence_with_model(input_file, save_path, max_length, model_path):
    """
    get data set and save to tfrecord
    :param data_dir:
    :return:
    """
    # load glove model
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
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
            if word in model:
                bytes_words.extend(model[word])
            else:
                bytes_words.extend([0] * 300)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":
                tf.train.Feature(int64_list=tf.train.Int64List(value=[label_list[index]])),
            "features":
                tf.train.Feature(float_list=tf.train.FloatList(value=bytes_words))
        }))
        writer.write(example.SerializeToString())


def save_word_ids(save_path, csv_path, glove_path, embedding_dim, seq_length, mode='train'):
    vocab, embd = loadGloVe(glove_path, embedding_dim)
    # init vocab processor
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(seq_length)
    # fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    lines = _read_csv(csv_path)
    split_lines = []
    label_list = []
    qid_list = []
    if mode == 'test':
        for line in lines:
            split_lines.append(' '.join(sentence_split(line[1], seq_length)))
            qid_list.append(str.encode(line[0]))
    else:
        for line in lines:
            split_lines.append(' '.join(sentence_split(line[1], seq_length)))
            label_list.append(int(line[2]))
            qid_list.append(str.encode(line[0]))
    word_ids = list(vocab_processor.transform(np.array(split_lines)))

    writer = tf.python_io.TFRecordWriter(save_path)

    if mode == 'test':
        for index, line in enumerate(word_ids):
            example = tf.train.Example(features=tf.train.Features(feature={
                "qid":
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[qid_list[index]])),
                "features":
                    tf.train.Feature(int64_list=tf.train.Int64List(value=line))
            }))
            writer.write(example.SerializeToString())
    else:
        for index, line in enumerate(word_ids):
            example = tf.train.Example(features=tf.train.Features(feature={
                "qid":
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[qid_list[index]])),
                "label":
                    tf.train.Feature(int64_list=tf.train.Int64List(value=[label_list[index]])),
                "features":
                    tf.train.Feature(int64_list=tf.train.Int64List(value=line))
            }))
            writer.write(example.SerializeToString())
    writer.close()


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
    max_length = 15
    dev_save_path = '../train_data/dev.tf_record'
    train_input_file = '../train_data/train.csv'
    train_save_path = '../train_data/train.tf_record'
    data_file = '../train_data/deal_train_data.csv'
    vocab_path = '../train_data/vocab.txt'
    dev_word_id_save_path = '../train_data/dev_word_id.tf_record'
    train_word_id_save_path = '../train_data/train_word_id.tf_record'
    # build_embedding_model(glove_file, gensim_file)
    # embedding_sentence(dev_input_file, dev_save_path, max_length)
    # embedding_sentence(train_input_file, train_save_path, max_length)
    # build_vocab(gensim_file, data_file, vocab_path)
    # embedding_sentence_with_model(dev_input_file, dev_save_path, max_length, gensim_file)
    # embedding_sentence_with_model(train_input_file, train_save_path, max_length, gensim_file)
    save_word_ids(dev_word_id_save_path, dev_input_file, vocab_path, embedding_dim, max_length)
    save_word_ids(train_word_id_save_path, train_input_file, vocab_path, embedding_dim, max_length)
