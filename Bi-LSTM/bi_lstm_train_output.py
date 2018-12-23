import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# from tensorflow import keras.keras.preprocessing.text.Tokenizer

tf.logging.set_verbosity(tf.logging.INFO)

# parameters config
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_data_path", "../train_data/train.csv", "train data path")
flags.DEFINE_string("dev_data_path", "../train_data/dev.csv", "dev data path")
flags.DEFINE_string("test_data_path", "../train_data/test.csv", "test data path")
flags.DEFINE_string("train_tfrecord_path", "../train_data/train_word_id.tf_record", "train data path")
flags.DEFINE_string("dev_tfrecord_path", "../train_data/dev_word_id.tf_record", "dev data path")
flags.DEFINE_string("test_tfrecord_path", "../train_data/test_word_id.tf_record", "test data path")
flags.DEFINE_integer("n_hidden", 128, "LSTM hidden layer num of features")
flags.DEFINE_integer("num_step", 16, "input data timesteps")
flags.DEFINE_integer("n_classes", 2, "number of classes")
flags.DEFINE_float("learning_rate", 0.01, "learnning rate")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("max_steps", 4000, "max step,stop condition")
flags.DEFINE_integer("display_step", 1000, "save model steps")
flags.DEFINE_string("train_writer_path", "./logs/train", "train tensorboard save path")
flags.DEFINE_string("dev_writer_path", "./logs/train", "dev tensorboard save path")
flags.DEFINE_string("checkpoint_path", "./logs/checkpoint", "model save path")
# flags.DEFINE_string("glove_path", "./glove.840B.300d/glove.840B.300d.txt", "pre-train embedding model path")
flags.DEFINE_string("glove_path", "../train_data/vocab.txt", "pre-train embedding model path")
flags.DEFINE_integer("embedding_dim", 300, "word embedding dim")
flags.DEFINE_integer("seq_length", 15, "sentence max length")


def re_build_data():
    # re-build data file
    train_data = pd.read_csv(FLAGS.raw_train_data_path)
    target_0_data = train_data.loc[train_data.target == 0, :]
    target_1_data = train_data.loc[train_data.target == 1, :]
    # view different target count
    print("target=0:%s" % len(target_0_data), "target=1:%s" % len(target_1_data))
    # 打乱数据集
    target_0_data = target_0_data.sample(frac=1.0)
    target_1_data = target_1_data.sample(frac=1.0)
    # 切分数据集
    target_0_train, target_0_test = target_0_data.iloc[:80000], target_0_data.iloc[80000:]
    target_1_train, target_1_test = target_1_data.iloc[:80000], target_1_data.iloc[80000:]
    # 合并训练数据并保存
    deal_train_data = target_0_train.append(target_1_train)
    deal_train_data = deal_train_data.sample(frac=1.0)
    deal_train_data.to_csv(FLAGS.deal_train_data_path, index=False)
    # build train data
    random_all_train_data = deal_train_data.sample(frac=1.0)
    # 13w for train and 3w for dev
    train_data, dev_data = random_all_train_data.iloc[:130000], random_all_train_data.iloc[130000:]
    return train_data[['question_text']].values, train_data[['target']].values, dev_data[['question_text']].values, \
           dev_data[['target']].values


def read_embedding_dict():
    """
    read embedding dictionary
    :return:
    """
    embedding_dict = {}
    with open(FLAGS.glove_path, 'r') as f:
        for line in tqdm(f):
            values = line.split(" ")
            word = values[0]
            value = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = value
    return embedding_dict


def text_to_array(text, embedding_dict):
    """
    Convert values to embeddings
    :param text:
    :param embedding_dict:
    :return:
    """
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:100]
    embeds = [embedding_dict.get(x, empyt_emb) for x in text]
    embeds += [empyt_emb] * (100 - len(embeds))
    return np.array(embeds)


def batch_gen(features, labels, epoch, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat(epoch).batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def model(n_hidden, input_data, weights, biases):
    """
    build bi-lstm model
    :param n_hidden:
    :param input_data:
    :param weights:
    :param biases:
    :return:
    """
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.7)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.7)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data, dtype=tf.float32)
    # 双向LSTM，输出outputs为两个cell的output
    # 将两个cell的outputs进行拼接
    outputs = tf.concat(outputs, 2)
    return tf.matmul(tf.transpose(outputs, [1, 0, 2])[-1], weights['out']) + biases['out']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
