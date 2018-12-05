import tensorflow as tf
import numpy as np


def loadGloVe(filename, emb_size):
    vocab = []
    embd = []
    vocab.append('unk')  # 装载不认识的词
    embd.append([0] * emb_size)  # 这个emb_size可能需要指定
    file = open(filename, 'r', encoding='utf-8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab, embd


def build_embedding_layer(vocab, embd):
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    return embedding_init, embedding, W, embedding_placeholder, vocab_size


def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0], [''], [0]]
    qid, question_text, target = tf.decode_csv(value, defaults)
    return question_text, target


def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(file_queue)
    # example = embedding(example)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    # example_batch = [tf.string_split(x) for x in example_batch]
    return example_batch, label_batch


def read_from_tfrecords(tfrecord_dir, batch_size):  # , max_length, embedding_dim
    """
    read data from tf_records
    TensorFlow基础5：TFRecords文件的存储与读取讲解及代码实现
    :param tfrecord_dir:
    :return:
    """
    # build file queue
    file_queue = tf.train.string_input_producer([tfrecord_dir])
    # build reader
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)

    features = tf.parse_single_example(value, features={
        "features": tf.FixedLenFeature([30], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    })

    label = features["label"]  # tf.cast(features["label"], tf.string)
    vector = features["features"]

    vector_batch, label_batch = tf.train.batch([vector, label], batch_size=batch_size, num_threads=4, capacity=32)
    # vector_batch shape is [batch_size,embedding_dim] and bi-lstm input must be [batch_size,max_time,depth]
    # max_time can be sentence max length and depth can be word embedding dimensions
    # reshape vector_batch
    # vector_batch = tf.reshape(vector, [batch_size, max_length, embedding_dim])

    return vector_batch, label_batch
