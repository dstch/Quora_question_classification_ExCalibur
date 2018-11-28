import tensorflow as tf
from .bi_lstm_model import bi_lstm

# parameters config
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("n_hidden", 256, "LSTM hidden layer num of features")
flags.DEFINE_integer("num_step", 32, "input data timesteps")
flags.DEFINE_integer("n_classes", 2, "number of classes")
flags.DEFINE_integer("learning_rate", 0.001, "learnning rate")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("max_steps", 4000, "max step,stop condition")

# tensorflow graph input
input_data = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_step], name='input data')
y = tf.placeholder("float", [None, FLAGS.n_classes])

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

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * FLAGS.batch_size < FLAGS.max_steps:
        pass
