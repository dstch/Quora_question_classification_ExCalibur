import tensorflow as tf
from data_helper import read_from_tfrecords, loadGloVe, build_embedding_layer, decode_array, embedding_raw_text
from bi_lstm_model import bi_lstm

tf.logging.set_verbosity(tf.logging.INFO)

# tensorflow graph input
input_data = tf.placeholder(dtype=tf.float32, shape=[None, 15, 300], name='input_data')

y = tf.placeholder("float", [None, 2])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'out': tf.Variable(tf.random_normal([2 * 128, 2]))
}
biases = {
    'out': tf.Variable(tf.random_normal([2]))
}

pred = bi_lstm().model2(128, input_data, weights, biases, 15, 300)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tfrecord_filename = '../train_data/dev_word_id.tf_record'
filename_queue = tf.train.string_input_producer([tfrecord_filename], )
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features={
    "features": tf.FixedLenFeature([15], tf.int64),
    "label": tf.FixedLenFeature([1], tf.int64)
})
label = tf.cast(features["label"], tf.int32)  # tf.cast(features["label"], tf.string)
vector = features["features"]

vector_batch, label_batch = tf.train.batch([vector, label], batch_size=32, num_threads=4, capacity=32)

indices = tf.expand_dims(tf.range(0, 32, 1), 1)
concated = tf.concat([indices, label_batch], 1)
onehot_labels = tf.sparse_to_dense(concated, tf.stack([32, 2]), 1.0, 0.0)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    vocab, embd = loadGloVe("../train_data/vocab.txt", 300)
    embedding_init, embedding, W, embedding_placeholder, vocab_size = build_embedding_layer(vocab, embd)
    W = sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
    vector_batch = tf.nn.embedding_lookup(W, vector_batch)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    step = 0
    while not coord.should_stop():
        example, l = sess.run([vector_batch, onehot_labels])
        tf.logging.info("start %s step optimizer" % step)
        sess.run(optimizer, feed_dict={
            input_data: example,
            y: l
        })
        if step % 1000 == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={input_data: example, y: l})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={input_data: example, y: l})
            tf.logging.info("Iter " + str(step) + ", Minibatch Loss= " + \
                            "{:.6f}".format(loss) + ", Training Accuracy= " + \
                            "{:.5f}".format(acc))
            print("Step:%s ,Testing Accuracy:" % step,
                  sess.run(accuracy, feed_dict={input_data: example, y: l}))
            saver.save(sess, './logs/checkpoint/model-%s' % step, global_step=step)
        step += 1
    coord.request_stop()
    coord.join(threads)
