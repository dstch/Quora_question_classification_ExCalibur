from collections import defaultdict

import tensorflow as tf
import gensim, jieba, string
import numpy as np

train_data_path = ''
test_data_path = ''
embedding_model_file = 'GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'


def get_stopwords():
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = []
        for word in f.readlines():
            stopwords.append(word.replace('\n', ''))
    return stopwords


def read_data(word_to_index, index_to_embedding, file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0], [''], [0]]
    qid, question_text, target = tf.decode_csv(value, defaults)
    batch_indexes=[]
    for w in question_text:
        batch_indexes.append(word_to_index[w.lower()])

    return question_text.split(), target


def embedding(data_list):
    vector_list = []
    stopwords = get_stopwords()
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_model_file, binary=True)
    for data in data_list:
        vector = np.array([0] * 300)
        for word in jieba.cut(data):
            if word not in stopwords and word not in string.punctuation and word.strip() != '':
                try:
                    vector = vector + model[word]
                except:
                    pass
        vector = vector / len(data.split())
        vector_list.append(tf.convert_to_tensor(vector))


def load_embedding_from_disks(with_indexes=True):
    """
    Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct
    `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
    else:
        word_to_embedding_dict = dict()
    glove_filename = 'glove.840B.300d/glove.840B.300d.txt'
    with open(glove_filename, 'r') as glove_file:
        for (i, line) in enumerate(glove_file):

            split = line.split(' ')

            word = split[0]

            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )

            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    _WORD_NOT_FOUND = [0.0] * len(representation)  # Empty representation for unknown words.
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        return word_to_embedding_dict


def create_pipeline(word_to_index, index_to_embedding, filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(word_to_index, index_to_embedding, file_queue)
    # example = embedding(example)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return example_batch, label_batch
