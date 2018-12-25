import functools
import tensorflow as tf
import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from tf_metrics import precision, recall, f1

# from tensorflow import keras.keras.preprocessing.text.Tokenizer

tf.logging.set_verbosity(tf.logging.INFO)

# parameters config
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("raw_train_data_path", "../raw_data/train.csv", "train data path")
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


def re_build_data(embedding_dict):
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
    target_0_train, target_0_test = target_0_data.iloc[:8000], target_0_data.iloc[8000:]
    target_1_train, target_1_test = target_1_data.iloc[:8000], target_1_data.iloc[8000:]
    # 合并训练数据并保存
    deal_train_data = target_0_train.append(target_1_train)
    deal_train_data = deal_train_data.sample(frac=1.0)
    # build train data
    random_all_train_data = deal_train_data.sample(frac=1.0)
    # 13w for train and 3w for dev
    train_data, dev_data = random_all_train_data.iloc[:13000], random_all_train_data.iloc[13000:]
    del random_all_train_data, deal_train_data, target_0_data, target_0_test, target_0_train, target_1_data, target_1_test, target_1_train
    return embedding_texts(train_data[['question_text']], embedding_dict), train_data[['target']].values, dev_data[
        ['question_text']].values, dev_data[['target']].values


def read_embedding_dict():
    """
    read embedding dictionary
    :return:
    """
    embedding_dict = {}
    with open(FLAGS.glove_path, 'r', encoding='utf-8') as f:
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
    text = text[:-1].split()[:20]
    embeds = [embedding_dict.get(x, empyt_emb) for x in text]
    embeds += [empyt_emb] * (20 - len(embeds))
    return np.array(embeds)


def embedding_texts(df, embedding_dict):
    return np.array([text_to_array(row[0], embedding_dict) for index, row in df.iterrows()])


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


def loadGloVe(filename, emb_size):
    vocab = []
    embd = []
    vocab.append('unk')  # 装载不认识的词
    embd.append([0] * emb_size)  # 这个emb_size可能需要指定
    file = open(filename, 'r', encoding='utf-8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append([float(x) for x in row[1:]])
    print('Loaded GloVe!')
    file.close()
    return vocab, embd


def batch_gen(features, labels, epoch, batch_size, n_classes):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(10).repeat(epoch).batch(batch_size)

    # Return the read end of the pipeline.
    (features_tensor, label_tensor) = dataset.make_one_shot_iterator().get_next()
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    label_tensor = tf.cast(label_tensor, tf.int32)
    concated = tf.concat([indices, label_tensor], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, n_classes]), 1.0, 0.0)
    return (features_tensor, onehot_labels)


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
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))


def input_fn(features, labels, params=None, shuffle_and_repeat=True):
    params = params if params is not None else {}

    split_lines = [' '.join(sentence_split(x, params['seq_length'])) for x in features]
    vocab, embd = loadGloVe(params['glove_path'], params['embedding_dim'])
    # init vocab processor
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(params['seq_length'])
    # fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    word_ids = np.array(list(vocab_processor.transform(np.array(split_lines))))

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((word_ids, labels))

    if shuffle_and_repeat:
        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(params.get('buffer', 1000)).repeat(params.get('epoch', 1)).batch(
            params.get('batch_size', 32))
    else:
        dataset = dataset.batch(params.get('batch_size', 32))
    return dataset


def model_fn(features, labels, mode, params):
    n_hidden = params['n_hidden']
    # dropout = params['dropout']

    vocab, embd = loadGloVe(params['glove_path'], params['embedding_dim'])
    W = np.array(embd)
    features = tf.nn.embedding_lookup(W, features, name='text_embedding')
    weights = {
        # Hidden layer weights => 2*n_hidden because of foward + backward cells
        'out': tf.Variable(tf.random_normal([2 * FLAGS.n_hidden, FLAGS.n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([FLAGS.n_classes]))
    }


    indices = tf.expand_dims(tf.range(0, params.get('batch_size', 32), 1), 1)
    label_tensor = tf.cast(labels, tf.int32)
    concated = tf.concat([indices, label_tensor], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([params.get('batch_size', 32), params.get('n_classes', 2)]),
                                       1.0, 0.0)

    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.7)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.7)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, features, dtype=tf.float32)
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
        indices = ['1', '0']
        # Loss
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=onehot_labels))
        metrics = {
            'acc': tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs, 1), tf.argmax(onehot_labels, 1)), tf.float32)),
            'precision': precision(onehot_labels, outputs, params.get('n_classes', 2), indices, weights['out']),
            'recall': recall(onehot_labels, outputs, params.get('n_classes', 2), indices, weights['out']),
            'f1': f1(onehot_labels, outputs, params.get('n_classes', 2), indices, weights['out']),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=cost, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params.get('learning_rate', 0.001)).minimize(cost)
            return tf.estimator.EstimatorSpec(
                mode, loss=cost, train_op=optimizer)


if __name__ == '__main__':
    # embedding_dict = {}  # read_embedding_dict()
    # train_text, train_target, _, _ = re_build_data(embedding_dict)
    # print(len(train_text), len(train_target))
    # batch_iterator = batch_gen(train_text, train_target, 1, 1, 2)
    # with tf.Session() as sess:
    #     for i in range(2):
    #         print(sess.run(batch_iterator))
    params = {
        'buffer': 1000,
        'epoch': 1,
        'batch_size': 32,
        'n_hidden': 128,
        'n_classes': 2,
        'learning_rate': 0.001,
        'glove_path': "../train_data/vocab.txt",
        'seq_length': 15,
        'embedding_dim': 300
    }
    # Estimator, train and evaluate
    train_data = pd.read_csv(FLAGS.train_data_path)
    dev_data = pd.read_csv(FLAGS.dev_data_path)
    train_inpf = functools.partial(input_fn, train_data['question_text'].values, train_data['target'].values, params)
    eval_inpf = functools.partial(input_fn, dev_data['question_text'].values, dev_data['target'].values, params)

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, './logs/results/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
