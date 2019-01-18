#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: bi_lstm_best.py
@time: 2019/1/17 10:19
@desc:
"""

import tensorflow as tf
import pandas as pd
from tensorflow.contrib.keras.api.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
import numpy as np
import string
import re

tf.logging.set_verbosity(tf.logging.INFO)

# parameters config
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_data_path", "../input/train.csv", "train data path")
flags.DEFINE_string("test_data_path", "../input/test.csv", "test data path")
flags.DEFINE_string("checkpoint_path", "./logs", "model save path")
flags.DEFINE_string("glove_path", "../input/embeddings/glove.840B.300d/glove.840B.300d.txt", "pre-train embedding model path")
flags.DEFINE_integer("max_sentence_len", 30, "max length of sentence")
flags.DEFINE_integer("embedding_dim", 300, "word embedding dim")
flags.DEFINE_integer("n_hidden", 128, "LSTM hidden layer num of features")
flags.DEFINE_integer("buffer_size", 1000, "batch size")
flags.DEFINE_integer("batch_size", 1000, "batch size")
flags.DEFINE_integer("n_classes", 2, "number of classes")
flags.DEFINE_float("learning_rate", 0.001, "learnning rate")
flags.DEFINE_integer("epoch",5,"train epoch")
flags.DEFINE_integer("attention_size", 128, "attention size")


def clean_punctuation(sentence):
    sentence = [x for x in sentence if x not in string.punctuation]
    sentence = ''.join(sentence)
    return sentence


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


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

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', ]


def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


mispell_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",
                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
                "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
                "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center',
                'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2',
                'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best',
                'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist',
                'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                'demonitization': 'demonetization', 'demonetisation': 'demonetization'}


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispellings, mispellings_re = _get_mispell(mispell_dict)


def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

# lower
train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())
test_df["question_text"] = test_df["question_text"].apply(lambda x: x.lower())

# Clean the text
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

# Clean numbers
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))

# Clean speelings
train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))

## fill up the missing values
train_X = train_df["question_text"].fillna("_##_").values
test_X = test_df["question_text"].fillna("_##_").values

# Get the response
train_y = train_df['target'].values
train_y = train_y.reshape(len(train_y), 1)


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
pred = model(FLAGS.n_hidden, embed, weights, biases, FLAGS.attention_size)

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


    sz = 30
    temp_y = train_y[:test_X.shape[0]]
    sub = test_df[['qid']]
    sess.run(test_init_op, feed_dict={X: test_X, Y: temp_y, batch_size: sz})
    sub['prediction'] = np.concatenate([sess.run(y_pred_cls) for _ in range(int(test_X.shape[0] / sz))])

    # sub['prediction'] = (sub['prediction'] > thresh).astype(np.int16)
    sub.to_csv("submission.csv", index=False)
    sub.sample()
