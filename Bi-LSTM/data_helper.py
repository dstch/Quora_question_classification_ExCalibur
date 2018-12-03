import tensorflow as tf
import numpy as np


def loadGloVe(filename, emb_size):
    vocab = []
    embd = []
    vocab.append('unk')  # 装载不认识的词
    embd.append([0] * emb_size)  # 这个emb_size可能需要指定
    file = open(filename, 'r')
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
    return embedding_init, embedding, W, embedding_placeholder,vocab_size


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
    example_batch = [tf.string_split(x) for x in example_batch]
    return example_batch, label_batch
