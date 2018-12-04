import csv
import tensorflow as tf
import gensim


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
        return lines


def sentence_split(sentence):
    """
    remove punctuation and split sentence.return list of words
    :param sentence:
    :return:
    """
    words = []
    return words


def embedding_sentence(model_file, input_file, embedding_dim, save_path):
    """
    get data set and save to tfrecord
    :param data_dir:
    :return:
    """
    lines = _read_csv(input_file)
    split_lines = []
    label_list = []
    for line in lines:
        split_lines.append(sentence_split(line[1]))
        label_list.append(line[2])
    del lines
    # load glove model
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)

    writer = tf.python_io.TFRecordWriter(save_path)
    for index, line in enumerate(split_lines):
        vector = [0] * embedding_dim
        for word in line:
            if word in model:
                vector = vector + model[word]
        vector = vector / len(line)
        example = tf.train.Example(features=tf.train.Feature(feature={
            "label":
                tf.train.Feature(int_list=tf.train.Int64List(value=[label_list[index]])),
            "features":
                tf.train.Feature(float_list=tf.train.FloatList(value=vector))
        }))
        writer.write(example.SerializeToString())
