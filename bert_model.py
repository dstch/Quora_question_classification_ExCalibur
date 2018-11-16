import tensorflow as tf


class bert_model():
    def __init__(self):
        self.pre_train_model_path = './raw_data/bert_model'
        self.pre_train_meta_graph = './raw_data/bert_model/bert_model.ckpt.meta'

    def load_pre_train_model(self):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.pre_train_meta_graph)
            saver.restore(sess, tf.train.latest_checkpoint(self.pre_train_model_path))
            graph = tf.get_default_graph()
            input1 = graph.get_tensor_by_name('')
            feed_dict = {}
