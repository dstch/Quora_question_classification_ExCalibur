import tensorflow as tf
import os
from .data_saver import save_word_ids
from .data_helper import read_from_tfrecords

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

# build test data tf_record
if os.path.exists(FLAGS.test_tfrecord_path) is not True:
    save_word_ids(FLAGS.test_tfrecord_path, FLAGS.test_data_path, FLAGS.glove_path, FLAGS.embedding_dim,
                  FLAGS.seq_length)

ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)  # 通过检查点文件锁定最新的模型
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)  # 载入参数，参数保存在两个文件中，不过restore会自己寻找
