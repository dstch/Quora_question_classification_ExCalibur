import tensorflow as tf
from bi_lstm_model import bi_lstm
from data_helper import read_from_tfrecords, loadGloVe, build_embedding_layer, calculate_evaluate_value
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.INFO)

# parameters config
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_data_path", "../train_data/train.csv", "train data path")
flags.DEFINE_string("test_data_path", "../train_data/dev.csv", "test data path")
flags.DEFINE_string("train_tfrecord_path", "../train_data/train_word_id.tf_record", "train data path")
flags.DEFINE_string("test_tfrecord_path", "../train_data/dev_word_id.tf_record", "test data path")
flags.DEFINE_integer("n_hidden", 128, "LSTM hidden layer num of features")
flags.DEFINE_integer("num_step", 16, "input data timesteps")
flags.DEFINE_integer("n_classes", 2, "number of classes")
flags.DEFINE_float("learning_rate", 0.01, "learnning rate")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("max_steps", 4000, "max step,stop condition")
flags.DEFINE_integer("display_step", 1000, "save model steps")
flags.DEFINE_string("train_writer_path", "./logs/train", "train tensorboard save path")
flags.DEFINE_string("test_writer_path", "./logs/train", "test tensorboard save path")
flags.DEFINE_string("checkpoint_path", "./logs/checkpoint", "model save path")
# flags.DEFINE_string("glove_path", "./glove.840B.300d/glove.840B.300d.txt", "pre-train embedding model path")
flags.DEFINE_string("glove_path", "../train_data/vocab.txt", "pre-train embedding model path")
flags.DEFINE_integer("embedding_dim", 300, "word embedding dim")
flags.DEFINE_integer("seq_length", 15, "sentence max length")

# tensorflow graph input
input_data = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.seq_length, FLAGS.embedding_dim], name='input_data')

y = tf.placeholder("float", [None, FLAGS.n_classes])

# get data batch
x_train_batch, y_train_batch = read_from_tfrecords(FLAGS.train_tfrecord_path, FLAGS.batch_size, FLAGS.seq_length,
                                                   FLAGS.n_classes, 2)
x_test, y_test = read_from_tfrecords(FLAGS.test_tfrecord_path, FLAGS.batch_size, FLAGS.seq_length,
                                     FLAGS.n_classes, 2)

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'out': tf.Variable(tf.random_normal([2 * FLAGS.n_hidden, FLAGS.n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([FLAGS.n_classes]))
}

pred = bi_lstm().model(FLAGS.n_hidden, input_data, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

# Evaluate model
y_pred_cls = tf.argmax(tf.nn.softmax(pred), 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# tensorboard
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()

# Initializing the variables
init = tf.initialize_all_variables()  # tf.global_variables_initializer()

tf.logging.info('Start Training...')

saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(FLAGS.train_writer_path, sess.graph)
    # test_writer = tf.summary.FileWriter(FLAGS.test_writer_path, sess.graph)

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    step = 1
    tf.logging.info("into embedding layer")
    # embedding layer
    vocab, embd = loadGloVe(FLAGS.glove_path, FLAGS.embedding_dim)
    embedding_init, embedding, W, embedding_placeholder, vocab_size = build_embedding_layer(vocab, embd)
    W = sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
    # embedding text
    x_train_batch = tf.nn.embedding_lookup(W, x_train_batch)
    x_test = tf.nn.embedding_lookup(W, x_test)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])
            # tf.logging.info("start %s step optimizer" % step)
            sess.run(optimizer, feed_dict={
                input_data: curr_x_train_batch,
                y: curr_y_train_batch
            })
            if step % FLAGS.display_step == 0:
                curr_x_test_batch, curr_y_test_batch = sess.run([x_test, y_test])  # shape(32,15)
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={input_data: curr_x_train_batch, y: curr_y_train_batch})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={input_data: curr_x_train_batch, y: curr_y_train_batch})
                tf.logging.info("Iter " + str(step) + ", Minibatch Loss= " + \
                                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                                "{:.5f}".format(acc))
                print("Step:%s ,Testing Accuracy:" % step,
                      sess.run(accuracy, feed_dict={input_data: curr_x_test_batch, y: curr_y_test_batch}))
                # save model
                saver.save(sess, FLAGS.checkpoint_path + '/model-%s' % step, global_step=step)
                # get prediction value
                pre = sess.run(y_pred_cls, feed_dict={input_data: curr_x_train_batch, y: curr_y_train_batch})
                # get real value
                y_true = curr_y_train_batch[:, 1]
                # calculate evaluate value
                tf_p, tf_r, tf_f1 = sess.run(calculate_evaluate_value(pre, y_true))
                print("prediction:%s   recall:%s   f1_score:%s" % (tf_p, tf_r, tf_f1))

                # evaluate by sklearn
                # 评估
                print("Precision, Recall and F1-Score...")
                print(metrics.classification_report(y_true, pre, target_names=['无意义', '有意义']))

                # 混淆矩阵
                print("Confusion Matrix...")
                cm = metrics.confusion_matrix(y_true, pre)
                print(cm)

                with tf.name_scope('Evaluation'):
                    tf.summary.scalar('prediction', tf_p)
                    tf.summary.scalar('recall', tf_r)
                    tf.summary.scalar('f1_score', tf_f1)
                summary_str = sess.run(merged_summary)
                train_writer = tf.summary.FileWriter(FLAGS.train_writer_path, sess.graph)
                train_writer.add_summary(summary_str, step)
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        tf.logging.info("Optimization Finished!")
        coord.request_stop()
    coord.join(threads)
