import functools
import tensorflow as tf
import pandas as pd
from tensorflow.contrib.keras.api.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
import numpy as np
import string, csv
from pathlib import Path

tf.logging.set_verbosity(tf.logging.INFO)

# parameters config
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_data_path", "../train_data/train.csv", "train data path")
flags.DEFINE_string("test_data_path", "../train_data/test.csv", "test data path")
flags.DEFINE_string("checkpoint_path", "./logs/checkpoint", "model save path")
flags.DEFINE_string("glove_path", "./glove.840B.300d/glove.840B.300d.txt", "pre-train embedding model path")
flags.DEFINE_integer("max_sentence_len", 15, "max length of sentence")


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
    train_data, dev_data = random_all_train_data.iloc[:130000], random_all_train_data.iloc[130000:]
    return train_data, dev_data


def clean_punctuation(sentence):
    sentence = [x for x in sentence if x not in string.punctuation]
    sentence = ''.join(sentence)
    return sentence


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def input_fn(features, labels, params=None, shuffle_and_repeat=True, isTest=False):
    params = params if params is not None else {}

    if isTest:
        labels_array = np.array([np.array(['0']) for x in features])
    else:
        labels_array = np.array([np.array([x]) for x in labels])
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels_array))

    if shuffle_and_repeat:
        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(params.get('buffer', 1000)).repeat(params.get('epoch', 1)).batch(
            params.get('batch_size', 32))
    else:
        dataset = dataset.batch(params.get('batch_size', 32))
    return dataset


def model_fn(features, labels, mode, params):
    n_hidden = params['n_hidden']
    embed = tf.nn.embedding_lookup(embeddings, features)
    # features = tf.cast(features, tf.float32)
    weights = {
        # Hidden layer weights => 2*n_hidden because of foward + backward cells
        'out': tf.Variable(tf.random_normal([2 * n_hidden, params.get('n_classes', 2)]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([params.get('n_classes', 2)]))
    }

    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.7)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.7)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embed, dtype=tf.float32)
    # 双向LSTM，输出outputs为两个cell的output
    # 将两个cell的outputs进行拼接
    outputs = tf.concat(outputs, 2)
    outputs = tf.matmul(tf.transpose(outputs, [1, 0, 2])[-1], weights['out']) + biases['out']
    pred = tf.argmax(tf.nn.softmax(outputs), 1)
    if mode == tf.estimator.ModeKeys.PREDICT:

        predictions = {
            'pred': pred
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        indices = tf.expand_dims(tf.range(0, params.get('batch_size', 32), 1), 1)
        label_tensor = tf.cast(labels, tf.int32)
        concated = tf.concat([indices, label_tensor], 1)
        onehot_labels = tf.sparse_to_dense(concated,
                                           tf.stack([params.get('batch_size', 32), params.get('n_classes', 2)]),
                                           1.0, 0.0)
        # Loss
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=onehot_labels))
        tags = tf.argmax(onehot_labels, 1)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred),
            'precision': tf.metrics.precision(tags, pred),
            'recall': tf.metrics.recall(tags, pred),
            'f1': tf.contrib.metrics.f1_score(tags, pred),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=cost, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params.get('learning_rate', 0.001)).minimize(cost,
                                                                                                          global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=cost, train_op=optimizer)


test_data = pd.read_csv(FLAGS.test_data_path)
train_data, dev_data = re_build_data()
# clean data
train_data["question_text"] = train_data["question_text"].map(lambda x: clean_punctuation(x))
test_data["question_text"] = test_data["question_text"].map(lambda x: clean_punctuation(x))
dev_data["question_text"] = dev_data["question_text"].map(lambda x: clean_punctuation(x))
# fill up the missing values
train_X = train_data["question_text"].fillna("_##_").values
val_X = dev_data["question_text"].fillna("_##_").values
test_X = test_data["question_text"].fillna("_##_").values

# creates a mapping from the words to the embedding vectors
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(FLAGS.glove_path, encoding='utf-8'))
vocab_size = len(embeddings_index.keys())
print('vocab size :', vocab_size)

tokenizer = Tokenizer(num_words=vocab_size, filters='', lower=False)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

train_X = pad_sequences(train_X, maxlen=FLAGS.max_sentence_len)
val_X = pad_sequences(val_X, maxlen=FLAGS.max_sentence_len)
test_X = pad_sequences(test_X, maxlen=FLAGS.max_sentence_len)

# Get the response
train_y = train_data['target'].values
val_y = dev_data['target'].values

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]
del all_embs

word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index))  # only want at most vocab_size words in our vocabulary
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

embeddings = tf.get_variable(name="embeddings", shape=embedding_matrix.shape,
                             initializer=tf.constant_initializer(np.array(embedding_matrix)),
                             trainable=False)

params = {
    'buffer': 128,
    'epoch': 10,
    'batch_size': 64,
    'n_hidden': 128,
    'n_classes': 2,
    'learning_rate': 0.001,
    'glove_path': "../train_data/vocab.txt",
    'seq_length': 15,
    'embedding_dim': 300,
    'embeddings': embeddings
}

train_inpf = functools.partial(input_fn, train_X, train_y, params)
eval_inpf = functools.partial(input_fn, val_X, val_y, params)
cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
estimator = tf.estimator.Estimator(model_fn, './logs/results/model', cfg, params)
Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
hook = tf.contrib.estimator.stop_if_no_increase_hook(
    estimator, 'acc', 500, min_steps=8000, run_every_secs=120)
train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
