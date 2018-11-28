import tensorflow as tf


class bi_lstm():
    def __init__(self):
        self.n_hidden = 256
        self.num_step = 32

    def model(self, FLAGS, weights, biases):
        input_data = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_step], name='input data')

        # forward and backward direction cell , add dropout layer
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.7)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.7)

        outputs, _, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data)

        return tf.matmul(outputs[-1], weights['out'] + biases['out'])
