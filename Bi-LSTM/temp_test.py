import tensorflow as tf
import operator
from tqdm import tqdm
import string
import pandas as pd
import numpy as np

# parameters config
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_data_path", "../train_data/train.csv", "train data path")
flags.DEFINE_string("test_data_path", "../train_data/test.csv", "test data path")
flags.DEFINE_string("checkpoint_path", "./logs/checkpoint", "model save path")
flags.DEFINE_string("glove_path", "./glove.840B.300d/glove.840B.300d.txt", "pre-train embedding model path")
flags.DEFINE_integer("max_sentence_len", 30, "max length of sentence")
flags.DEFINE_integer("embedding_dim", 300, "word embedding dim")
flags.DEFINE_integer("n_hidden", 256, "LSTM hidden layer num of features")
flags.DEFINE_integer("batch_size", 1000, "batch size")
flags.DEFINE_integer("buffer_size", 1000, "batch size")
flags.DEFINE_integer("n_classes", 2, "number of classes")
flags.DEFINE_float("learning_rate", 0.001, "learnning rate")
flags.DEFINE_float("epoch", 10, "epoch of train")
flags.DEFINE_integer("attention_size", 256, "attention size")


def build_vocab(sentences, verbose=True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass
    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


def clean_punctuation(sentence):
    sentence = [x for x in sentence if x not in string.punctuation]
    sentence = ''.join(sentence)
    return sentence


def model1(n_hidden, input_data, weights, biases):
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.7)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.7)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data, dtype=tf.float32)
    # 双向LSTM，输出outputs为两个cell的output
    # 将两个cell的outputs进行拼接
    outputs = tf.concat(outputs, 2)
    return tf.matmul(tf.transpose(outputs, [1, 0, 2])[-1], weights['out']) + biases['out']


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def model(n_hidden, input_data, weights, biases, attention_size):
    # bi-lstm
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.7)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.7)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data, dtype=tf.float32)
    outputs = tf.concat(outputs, 2)
    outputs = tf.transpose(outputs, [1, 0, 2])
    # attention layer
    with tf.name_scope('attention'), tf.variable_scope('attention'):
        attention_w = tf.Variable(tf.truncated_normal([2 * n_hidden, attention_size], stddev=0.1), name='attention_w')
        attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
        u_list = []
        for t in range(FLAGS.max_sentence_len):
            u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b)
            u_list.append(u_t)
        u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
        attn_z = []
        for t in range(FLAGS.max_sentence_len):
            z_t = tf.matmul(u_list[t], u_w)
            attn_z.append(z_t)
        # transform to batch_size * sequence_length
        attn_zconcat = tf.concat(attn_z, axis=1)
        alpha = tf.nn.softmax(attn_zconcat)
        # transform to sequence_length * batch_size * 1 , same rank as outputs
        alpha_trans = tf.reshape(tf.transpose(alpha, [1, 0]), [FLAGS.max_sentence_len, -1, 1])
        final_output = tf.reduce_sum(outputs * alpha_trans, 0)

    # 双向LSTM，输出outputs为两个cell的output
    # 将两个cell的outputs进行拼接
    # outputs = tf.concat(outputs, 2)
    # return tf.matmul(tf.transpose(outputs, [1, 0, 2])[-1], weights['out']) + biases['out']
    return tf.matmul(final_output, weights['out']) + biases['out']


# input_data = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.max_sentence_len, FLAGS.embedding_dim],
#                             name='input_data')
#
# # Define weights
# weights = {
#     # Hidden layer weights => 2*n_hidden because of foward + backward cells
#     'out': tf.Variable(tf.random_normal([2 * FLAGS.n_hidden, FLAGS.n_classes]))
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([FLAGS.n_classes]))
# }
# pred = model(FLAGS.n_hidden, input_data, weights, biases, FLAGS.attention_size)
# print(pred)

# train_data = pd.read_csv(FLAGS.train_data_path)
# # clean data
# sentences = train_data["question_text"].map(lambda x: clean_punctuation(x))
# vocab = build_vocab(sentences)
# embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(FLAGS.glove_path, encoding='utf-8'))
# oov = check_coverage(vocab,embeddings_index)

# global_step = tf.Variable(0, trainable=False)
#
# init = tf.group(tf.initialize_all_variables(), tf.local_variables_initializer())
# with tf.Session() as sess:
#     sess.run(init)
#     for step in range(1, 5):
#         assign_op = global_step.assign(step)
#         print(sess.run())

t1=tf.Variable(0, trainable=False)
print(t1)