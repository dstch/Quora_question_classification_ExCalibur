import tensorflow as tf
import numpy as np
import csv, string


def _read_csv(input_file, max_length):
    """
    read csv file,get data
    :param input_file:
    :return:
    """
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f)
        text_lines = []
        # label_lines = []
        for line in reader:
            text_lines.append(' '.join(sentence_split(line[1], max_length)))
            # label_lines.append(int(line[2]))
        return text_lines[1:]  # remove header


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


def read_from_tfrecords(tfrecord_dir, batch_size, max_length, n_classes, epochs):  # , max_length, embedding_dim
    """
    read data from tf_records
    TensorFlow基础5：TFRecords文件的存储与读取讲解及代码实现
    :param tfrecord_dir:
    :return:
    """
    # build file queue
    file_queue = tf.train.string_input_producer([tfrecord_dir], num_epochs=epochs)
    # build reader
    reader = tf.TFRecordReader()
    _, value = reader.read(file_queue)

    features = tf.parse_single_example(value, features={
        "features": tf.FixedLenFeature([max_length], tf.int64),
        "label": tf.FixedLenFeature([1], tf.int64)
    })

    label = tf.cast(features["label"], tf.int32)  # tf.cast(features["label"], tf.string)
    vector = features["features"]

    vector_batch, label_batch = tf.train.batch([vector, label], batch_size=batch_size, num_threads=4, capacity=256)

    # deal with label batch, change int label to one-hot code
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, label_batch], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, n_classes]), 1.0, 0.0)

    return vector_batch, onehot_labels


def decode_array(array):
    decode_array = []
    for line in array:
        temp_line = []
        for word in line:
            temp_line.append(word.decode())
        decode_array.append(' '.join(temp_line))
    return np.array(decode_array)


def embedding_raw_text(csv_path, max_length, W, vocab_processor, batch_size, n_classes):
    lines = _read_csv(csv_path, max_length)
    text = tf.nn.embedding_lookup(W, np.array(list(vocab_processor.transform(np.array(lines)))))
    file_queue = tf.train.string_input_producer([csv_path], num_epochs=None)
    _, label = read_data(file_queue)
    label = tf.reshape(label, (1,))
    vector_batch, label_batch = tf.train.batch([text, label], batch_size=batch_size, num_threads=4, capacity=32)

    # deal with label batch, change int label to one-hot code
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, label_batch], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, n_classes]), 1.0, 0.0)

    return vector_batch, onehot_labels
