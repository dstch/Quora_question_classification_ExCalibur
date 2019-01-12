import tensorflow as tf
import pandas as pd
from tensorflow.contrib.keras.api.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
import numpy as np
import string

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


def re_build_data():
    # re-build data file
    train_data = pd.read_csv(FLAGS.train_data_path)
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
    # build train data
    random_all_train_data = deal_train_data.sample(frac=1.0)
    # 13w for train and 3w for dev
    # train_data, dev_data = random_all_train_data.iloc[:130000], random_all_train_data.iloc[130000:]
    # return train_data, dev_data
    return random_all_train_data


def clean_punctuation(sentence):
    sentence = [x for x in sentence if x not in string.punctuation]
    sentence = ''.join(sentence)
    return sentence


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def model(n_hidden, input_data, weights, biases):
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.7)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.7)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data, dtype=tf.float32)
    # 双向LSTM，输出outputs为两个cell的output
    # 将两个cell的outputs进行拼接
    outputs = tf.concat(outputs, 2)
    return tf.matmul(tf.transpose(outputs, [1, 0, 2])[-1], weights['out']) + biases['out']


test_data = pd.read_csv(FLAGS.test_data_path)
# train_data, dev_data = re_build_data()
# train_data = re_build_data()
train_data = pd.read_csv(FLAGS.train_data_path)
# clean data
train_data["question_text"] = train_data["question_text"].map(lambda x: clean_punctuation(x))
test_data["question_text"] = test_data["question_text"].map(lambda x: clean_punctuation(x))
# dev_data["question_text"] = dev_data["question_text"].map(lambda x: clean_punctuation(x))
# Get the response
train_y = train_data['target'].values
# val_y = dev_data['target'].values
train_y = train_y.reshape(len(train_y), 1)
# val_y = val_y.reshape(len(val_y), 1)
# fill up the missing values
train_X = train_data["question_text"].fillna("_##_").values
# val_X = dev_data["question_text"].fillna("_##_").values
test_X = test_data["question_text"].fillna("_##_").values

# creates a mapping from the words to the embedding vectors=
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(FLAGS.glove_path, encoding='utf-8'))
vocab_size = len(embeddings_index.keys())
print('vocab size :', vocab_size)

tokenizer = Tokenizer(num_words=vocab_size, filters='', lower=False)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
# val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

train_X = pad_sequences(train_X, maxlen=FLAGS.max_sentence_len)
# val_X = pad_sequences(val_X, maxlen=FLAGS.max_sentence_len)
test_X = pad_sequences(test_X, maxlen=FLAGS.max_sentence_len)

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]
del all_embs

word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index)) + 1  # only want at most vocab_size words in our vocabulary
embedding_matrix = np.random.normal(emb_mean, emb_std,
                                    (nb_words, embed_size))

for word, i in word_index.items():  # insert embeddings we that exist into our matrix
    if i >= vocab_size:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        try:
            embedding_matrix[i] = embedding_vector
        except:
            print('error:', i)
            print(embedding_matrix.shape)

X = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='X')
Y = tf.placeholder(tf.int32, [None, 1], name='Y')
batch_size = tf.placeholder(tf.int64)

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size)).shuffle(
    buffer_size=FLAGS.buffer_size).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)

iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

train_init_op = iterator.make_initializer(dataset)
test_init_op = iterator.make_initializer(test_dataset)

questions, labels = iterator.get_next()

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
pred = model(FLAGS.n_hidden, embed, weights, biases)

indices = tf.expand_dims(tf.range(0, FLAGS.batch_size, 1), 1)
concated = tf.concat([indices, labels], 1)
labels = tf.sparse_to_dense(concated, tf.stack([FLAGS.batch_size, FLAGS.n_classes]), 1.0, 0.0)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

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
        prev_iter = 0.
        for i in range(num_batches):
            _, batch_loss, _ = sess.run([optimizer, cost, f1_update])
            iter_cost += batch_loss

            # End training after

            if (i % num_iter == 0 and i > 0):
                iter_cost /= (i - prev_iter)  # get average batch cost
                prev_iter = i  # update prev_iter for next iteration
                cur_f1 = sess.run(F1)
                sess.run(reset_op)  # reset counters for F1

                f1.append(cur_f1)
                costs.append(iter_cost)
                print("Epoch %s Iteration %s cost: %s  f1: %s " % (epoch, i, iter_cost, cur_f1))
                batch_cost = 0.  # reset batch_cost)

    # bs = 100
    # sess.run(test_init_op, feed_dict={X: val_X, Y: val_y, batch_size: bs})
    # val_cost = 0.
    # num_batches = int(val_X.shape[0] / bs)  # number of minibatches of size minibatch_size in the train set
    # tf.set_random_seed(2018)
    #
    # for _ in range(num_batches):
    #     sess.run(f1_update)
    # print('Validation f1: ', sess.run(F1))

    sz = 30
    temp_y = train_y[:test_X.shape[0]]
    sub = test_data[['qid']]
    sess.run(test_init_op, feed_dict={X: test_X, Y: temp_y, batch_size: sz})
    sub['prediction'] = np.concatenate([sess.run(y_pred_cls) for _ in range(int(test_X.shape[0] / sz))])

    # sub['prediction'] = (sub['prediction'] > thresh).astype(np.int16)
    sub.to_csv("submission.csv", index=False)
    sub.sample()
