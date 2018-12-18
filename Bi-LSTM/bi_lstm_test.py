import tensorflow as tf
import os
from data_saver import save_word_ids
from data_helper import read_from_tfrecords, loadGloVe, build_embedding_layer
import csv
from bi_lstm_model import bi_lstm

tf.logging.set_verbosity(tf.logging.INFO)

# parameters config
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", "./logs/checkpoint", "model save path")
flags.DEFINE_string("test_data_path", "../train_data/test.csv", "test data path")
flags.DEFINE_string("test_tfrecord_path", "../train_data/test_word_id.tf_record", "test data path")
flags.DEFINE_string("glove_path", "../train_data/vocab.txt", "pre-train embedding model path")
flags.DEFINE_integer("embedding_dim", 300, "word embedding dim")
flags.DEFINE_integer("seq_length", 15, "sentence max length")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("n_classes", 2, "number of classes")
flags.DEFINE_string("submit_file", "../train_data/submit.csv", "prediciton result file")
flags.DEFINE_integer("n_hidden", 128, "LSTM hidden layer num of features")

# build test data tf_record
if os.path.exists(FLAGS.test_tfrecord_path) is not True:
    save_word_ids(FLAGS.test_tfrecord_path, FLAGS.test_data_path, FLAGS.glove_path, FLAGS.embedding_dim,
                  FLAGS.seq_length, 'test')

with tf.Graph().as_default():
    # tensorflow graph input
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.seq_length, FLAGS.embedding_dim],
                                name='input_data')
    # get test data
    test_text_batch, test_qid_batch = read_from_tfrecords(FLAGS.test_tfrecord_path, FLAGS.batch_size, FLAGS.seq_length,
                                                          FLAGS.n_classes, 1, 'test')
    # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)  # 通过检查点文件锁定最新的模型
    # saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
    # Define weights
    weights = {
        # Hidden layer weights => 2*n_hidden because of foward + backward cells
        'out': tf.Variable(tf.random_normal([2 * FLAGS.n_hidden, FLAGS.n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([FLAGS.n_classes]))
    }
    pred = bi_lstm().model(FLAGS.n_hidden, input_data, weights, biases)
    logit = tf.argmax(tf.nn.softmax(pred), 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())
        sess.run(tf.local_variables_initializer())

        # embedding layer
        vocab, embd = loadGloVe(FLAGS.glove_path, FLAGS.embedding_dim)
        embedding_init, embedding, W, embedding_placeholder, vocab_size = build_embedding_layer(vocab, embd)
        W = sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
        # embedding text
        test_text_batch = tf.nn.embedding_lookup(W, test_text_batch, name='train_text_embedding')

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # 载入参数，参数保存在两个文件中，不过restore会自己寻找
        else:
            tf.logging.ERROR("Load model failed, can't find model")
            raise FileNotFoundError
        # graph = tf.get_default_graph()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        save_qid_list = []
        save_target_list = []
        try:
            while not coord.should_stop():
                test_batch_value, qid = sess.run([test_text_batch, test_qid_batch])
                pred_value = sess.run(logit, feed_dict={input_data: test_batch_value})
                save_qid_list.extend(qid)
                save_target_list.extend(pred_value)
        except tf.errors.OutOfRangeError:
            print("Done testing -- epoch limit reached")
        finally:
            coord.request_stop()
        coord.join(threads)
        # write result to csv
        headers = ['qid', 'target']
        save_data_list = []
        for index, qid in enumerate(save_qid_list):
            qid_str = qid[0].decode()
            save_data_list.append([qid_str, save_target_list[index]])
        with open(FLAGS.submit_file, 'w+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(save_data_list)
