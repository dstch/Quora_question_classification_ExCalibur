import tensorflow as tf


# https://cuiqingcai.com/5001.html
class bi_lstm():
    def __init__(self):
        self.n_hidden = 256
        self.num_step = 32

    def model(self, FLAGS, input_data, weights, biases, n_steps, n_input):
        # x = tf.transpose(input_data, [1, 0, 2])
        # x = tf.reshape(x, [-1, n_input])
        # x = tf.split(x, n_steps)
        # forward and backward direction cell , add dropout layer
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.7)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.7)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data, dtype=tf.float32)
        # 双向LSTM，输出outputs为两个cell的output
        # 将两个cell的outputs进行拼接
        outputs = tf.concat(outputs, 2)
        return tf.matmul(tf.transpose(outputs, [1, 0, 2])[-1], weights['out']) + biases['out']

    def model2(self, n_hidden, input_data, weights, biases, n_steps, n_input):
        # x = tf.transpose(input_data, [1, 0, 2])
        # x = tf.reshape(x, [-1, n_input])
        # x = tf.split(x, n_steps)
        # forward and backward direction cell , add dropout layer
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.7)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.7)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data, dtype=tf.float32)
        # 双向LSTM，输出outputs为两个cell的output
        # 将两个cell的outputs进行拼接
        outputs = tf.concat(outputs, 2)
        return tf.matmul(tf.transpose(outputs, [1, 0, 2])[-1], weights['out']) + biases['out']
