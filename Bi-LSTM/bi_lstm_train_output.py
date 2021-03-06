import functools
import tensorflow as tf
import pandas as pd
import numpy as np
import string, csv
from pathlib import Path
import gensim

tf.logging.set_verbosity(tf.logging.INFO)

# parameters config
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("raw_train_data_path", "../train_data/raw_train.csv", "train data path")
flags.DEFINE_string("train_data_path", "../train_data/train.csv", "train data path")
flags.DEFINE_string("dev_data_path", "../train_data/dev.csv", "dev data path")
flags.DEFINE_string("deal_train_data_path", "../train_data/deal_train.csv", "")
flags.DEFINE_string("test_data_path", "../train_data/test.csv", "test data path")
flags.DEFINE_string("checkpoint_path", "./logs/checkpoint", "model save path")
# flags.DEFINE_string("glove_path", "./glove.840B.300d/glove.840B.300d.txt", "pre-train embedding model path")
flags.DEFINE_string("glove_path", "../train_data/vocab.txt", "pre-train embedding model path")
flags.DEFINE_string("gensim_path", "./glove.840B.300d/glove_model.txt", "")


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
    # train_data.to_csv(FLAGS.train_data_path, index=False)
    # dev_data.to_csv(FLAGS.dev_data_path, index=False)
    return train_data, dev_data


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


# read csv
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


# build vocab
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


def generator_fn(words, qid):
    for line_words, line_tags in zip(words, qid):
        yield (line_words, line_tags)


def input_fn(features, labels, params=None, shuffle_and_repeat=True, isTest=False):
    params = params if params is not None else {}

    split_lines = [' '.join(sentence_split(x, params['seq_length'])) for x in features]
    if isTest:
        labels_array = np.array([np.array(['0']) for x in features])
    else:
        labels_array = np.array([np.array([x]) for x in labels])
    vocab, embd = loadGloVe(params['glove_path'], params['embedding_dim'])
    # init vocab processor
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(params['seq_length'])
    # fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    word_ids = np.array(list(vocab_processor.transform(np.array(split_lines))))

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((word_ids, labels_array))

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
    features = tf.cast(features, tf.float32)
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


if __name__ == '__main__':
    train_data, dev_data = re_build_data()
    # build_vocab(FLAGS.gensim_path, FLAGS.deal_train_data_path, FLAGS.glove_path)
    params = {
        'buffer': 128,
        'epoch': 10,
        'batch_size': 64,
        'n_hidden': 128,
        'n_classes': 2,
        'learning_rate': 0.001,
        'glove_path': "../train_data/vocab.txt",
        'seq_length': 15,
        'embedding_dim': 300
    }
    # Estimator, train and evaluate
    # train_data = pd.read_csv(FLAGS.train_data_path)
    # dev_data = pd.read_csv(FLAGS.dev_data_path)
    train_inpf = functools.partial(input_fn, train_data['question_text'].values, train_data['target'].values, params)
    eval_inpf = functools.partial(input_fn, dev_data['question_text'].values, dev_data['target'].values, params)

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, './logs/results/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'acc', 500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # Estimator, predict
    test_data = pd.read_csv(FLAGS.test_data_path)
    test_inpf = functools.partial(input_fn, test_data['question_text'].values, [], params, shuffle_and_repeat=False,
                                  isTest=True)
    text_gen = generator_fn(test_data['question_text'].values, test_data['qid'].values)
    preds_gen = estimator.predict(test_inpf)
    save_data = [['qid', 'target']]
    for texts, preds in zip(text_gen, preds_gen):
        part_save_data = []
        (words, qid) = texts
        save_data.append([qid, preds['pred']])
    pd.DataFrame(save_data).to_csv('./logs/results/submission.csv.csv', index=False, header=False)
