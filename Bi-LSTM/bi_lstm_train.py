import tensorflow as tf
from .bi_lstm_model import bi_lstm
from .data_helper import create_pipeline, build_embedding_layer, loadGloVe
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

# parameters config
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_data_path", "../train_data/train.csv", "train data path")
flags.DEFINE_string("test_data_path", "../train_data/dev.csv", "test data path")
flags.DEFINE_integer("n_hidden", 256, "LSTM hidden layer num of features")
flags.DEFINE_integer("num_step", 32, "input data timesteps")
flags.DEFINE_integer("n_classes", 2, "number of classes")
flags.DEFINE_float("learning_rate", 0.001, "learnning rate")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("max_steps", 4000, "max step,stop condition")
flags.DEFINE_integer("display_step", 1000, "save model steps")
flags.DEFINE_string("train_writer_path", "./logs/train", "train tensorboard save path")
flags.DEFINE_string("test_writer_path", "./logs/train", "test tensorboard save path")
flags.DEFINE_string("checkpoint_path", "./logs/checkpoint", "model save path")
flags.DEFINE_string("glove_path", "../glove.840B.300d/glove.840B.300d.txt", "pre-train embedding model path")
flags.DEFINE_integer("embedding_dim", 300, "word embedding dim")

# tensorflow graph input
input_data = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_step], name='input_data')
y = tf.placeholder("float", [None, FLAGS.n_classes])

# get data batch
x_train_batch, y_train_batch = create_pipeline(FLAGS.train_data_path,
                                               FLAGS.batch_size, num_epochs=FLAGS.max_steps)
x_test, y_test = create_pipeline(FLAGS.test_data_path, FLAGS.batch_size)

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'out': tf.Variable(tf.random_normal([2 * FLAGS.n_hidden, FLAGS.n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([FLAGS.n_classes]))
}

pred = bi_lstm().model(FLAGS, input_data, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# tensorboard
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()

tf.logging.info('Start Training...')

saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(FLAGS.train_writer_path, sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.test_writer_path, sess.graph)

    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 1

    # embedding layer
    vocab, embd = loadGloVe(FLAGS.glove_path, FLAGS.embedding_dim)
    embedding_init, embedding, W, embedding_placeholder, vocab_size = build_embedding_layer(vocab, embd)
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

    # init vocab processor
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(vocab_size)
    # fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    # transform inputs
    x_train_batch = tf.nn.embedding_lookup(W, np.array(list(vocab_processor.transform(x_train_batch))))
    x_test = tf.nn.embedding_lookup(W, np.array(list(vocab_processor.transform(x_test))))

    curr_x_test_batch, curr_y_test_batch = sess.run([x_test, y_test])
    while not coord.should_stop():
        curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])

        sess.run(optimizer, feed_dict={
            input_data: curr_x_train_batch,
            y: curr_y_train_batch
        })
        if step % FLAGS.display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={input_data: curr_x_train_batch, y: curr_y_train_batch})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={input_data: curr_x_train_batch, y: curr_y_train_batch})
            tf.logging.info("Iter " + str(step) + ", Minibatch Loss= " + \
                            "{:.6f}".format(loss) + ", Training Accuracy= " + \
                            "{:.5f}".format(acc))
            tf.logging.info("Step:%d ,Testing Accuracy:" % step,
                            sess.run(accuracy, feed_dict={input_data: curr_x_test_batch, y: curr_y_test_batch}))
            saver.save(sess, FLAGS.checkpoint_path, global_step=step)
        step += 1

    tf.logging.info("Optimization Finished!")
    coord.request_stop()
    coord.join(threads)
