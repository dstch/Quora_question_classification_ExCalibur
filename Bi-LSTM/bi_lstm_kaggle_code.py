import tensorflow as tf
import pandas as pd
from tensorflow.contrib.keras.api.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
import numpy as np
import string
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context

tf.logging.set_verbosity(tf.logging.INFO)

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


def clean_punctuation(sentence):
    sentence = [x for x in sentence if x not in string.punctuation]
    sentence = ''.join(sentence)
    return sentence


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def cyclic_learning_rate(global_step,
                         learning_rate=0.01,
                         max_lr=0.002,
                         step_size=3000.,
                         gamma=0.99994,
                         mode='exp_range',
                         name=None):
    if global_step is None:
        raise ValueError("global_step is required for cyclic_learning_rate.")
    with ops.name_scope(name, "CyclicLearningRate",
                        [learning_rate, global_step]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(step_size, dtype)

        def cyclic_lr():
            """Helper to recompute learning rate; most helpful in eager-mode."""
            # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
            double_step = math_ops.multiply(2., step_size)
            global_div_double_step = math_ops.divide(global_step, double_step)
            cycle = math_ops.floor(math_ops.add(1., global_div_double_step))
            # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
            double_cycle = math_ops.multiply(2., cycle)
            global_div_step = math_ops.divide(global_step, step_size)
            tmp = math_ops.subtract(global_div_step, double_cycle)
            x = math_ops.abs(math_ops.add(1., tmp))
            # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
            a1 = math_ops.maximum(0., math_ops.subtract(1., x))
            a2 = math_ops.subtract(max_lr, learning_rate)
            clr = math_ops.multiply(a1, a2)
            if mode == 'triangular2':
                clr = math_ops.divide(clr, math_ops.cast(math_ops.pow(2, math_ops.cast(
                    cycle - 1, tf.int32)), tf.float32))
            if mode == 'exp_range':
                clr = math_ops.multiply(math_ops.pow(gamma, global_step), clr)
            return math_ops.add(clr, learning_rate, name=name)

        if not context.executing_eagerly():
            cyclic_lr = cyclic_lr()

    return cyclic_lr


def model(n_hidden, input_data, weights, biases, attention_size):
    # bi-lstm
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.7)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.7)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data, dtype=tf.float32)
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


test_data = pd.read_csv(FLAGS.test_data_path)
train_data = pd.read_csv(FLAGS.train_data_path)
# clean data
train_data["question_text"] = train_data["question_text"].map(lambda x: clean_punctuation(x))
test_data["question_text"] = test_data["question_text"].map(lambda x: clean_punctuation(x))
# fill up the missing values
train_X = train_data["question_text"].fillna("_##_").values
test_X = test_data["question_text"].fillna("_##_").values
# creates a mapping from the words to the embedding vectors
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(FLAGS.glove_path, encoding='utf-8'))
vocab_size = len(embeddings_index.keys())
print('vocab size :', vocab_size)

tokenizer = Tokenizer(num_words=vocab_size, filters='', lower=False)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)

train_X = pad_sequences(train_X, maxlen=FLAGS.max_sentence_len)
test_X = pad_sequences(test_X, maxlen=FLAGS.max_sentence_len)

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]
del all_embs

word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index)) + 1  # only want at most vocab_size words in our vocabulary
embedding_matrix = np.random.normal(emb_mean, emb_std,
                                    (nb_words, embed_size))
# insert embeddings we that exist into our matrix
for word, i in word_index.items():
    if i >= vocab_size:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Get the labels
train_y = train_data['target'].values
train_y = train_y.reshape(len(train_y), 1)

# input tensor
X = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='X')
Y = tf.placeholder(tf.int32, [None, 1], name='Y')
batch_size = tf.placeholder(tf.int64)

# build dataset and batch
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size)).shuffle(
    buffer_size=FLAGS.buffer_size).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)

iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

train_init_op = iterator.make_initializer(dataset)
test_init_op = iterator.make_initializer(test_dataset)

questions, labels = iterator.get_next()

# embedding layer
embeddings = tf.get_variable(name="embeddings", shape=embedding_matrix.shape,
                             initializer=tf.constant_initializer(np.array(embedding_matrix)),
                             trainable=False)

embed = tf.nn.embedding_lookup(embeddings, questions)

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'out': tf.Variable(tf.random_normal([2 * FLAGS.n_hidden, FLAGS.n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([FLAGS.n_classes]))
}
pred = model(FLAGS.n_hidden, embed, weights, biases, FLAGS.attention_size)

indices = tf.expand_dims(tf.range(0, FLAGS.batch_size, 1), 1)
concated = tf.concat([indices, labels], 1)
labels = tf.sparse_to_dense(concated, tf.stack([FLAGS.batch_size, FLAGS.n_classes]), 1.0, 0.0)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
global_step = tf.identity(tf.Variable(0, trainable=False))
# optimizer = tf.train.AdamOptimizer(
#     learning_rate=cyclic_learning_rate(global_step, learning_rate=FLAGS.learning_rate)).minimize(cost)
optimizer = tf.train.AdamOptimizer(
    learning_rate=tf.train.cosine_decay(FLAGS.learning_rate, global_step, 3000)).minimize(cost)

# Evaluate model
tags = tf.argmax(labels, 1)
y_pred_cls = tf.argmax(tf.nn.softmax(pred), 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tags)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.name_scope('metrics'):
    F1, f1_update = tf.contrib.metrics.f1_score(labels=tags, predictions=y_pred_cls, name='my_metric')

running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
reset_op = tf.variables_initializer(var_list=running_vars)

init = tf.group(tf.initialize_all_variables(), tf.local_variables_initializer())

f1, costs = [], []

with tf.Session() as sess:
    sess.run(init)
    num_iter = 1000
    data = sess.run(train_init_op, feed_dict={X: train_X, Y: train_y, batch_size: FLAGS.batch_size})
    num_batches = int(train_X.shape[0] / FLAGS.batch_size)
    for epoch in range(FLAGS.epoch):
        iter_cost = 0.
        for i in range(num_batches):
            _, batch_loss, _ = sess.run([optimizer, cost, f1_update], feed_dict={global_step: i})
            iter_cost += batch_loss

            # End training after
            if (i % num_iter == 0 and i > 0):
                cur_f1 = sess.run(F1)
                sess.run(reset_op)  # reset counters for F1

                f1.append(cur_f1)
                costs.append(iter_cost)
                print("Epoch %s Iteration %s cost: %s  f1: %s " % (epoch, i, iter_cost, cur_f1))
                batch_cost = 0.  # reset batch_cost)

    sz = 30
    temp_y = train_y[:test_X.shape[0]]
    sub = test_data[['qid']]
    sess.run(test_init_op, feed_dict={X: test_X, Y: temp_y, batch_size: sz})
    sub['prediction'] = np.concatenate([sess.run(y_pred_cls) for _ in range(int(test_X.shape[0] / sz))])

    # sub['prediction'] = (sub['prediction'] > thresh).astype(np.int16)
    sub.to_csv("submission.csv", index=False)
    sub.sample()
