import numpy as np
import tensorflow as tf
import os
import utils
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


flags = tf.app.flags
flags.DEFINE_string('data_dir', './datasets', 'Data dir path.')
flags.DEFINE_string('dataset', 'imdb', 'Data dir path.')
flags.DEFINE_string('checkpoint_dir', './ckpt', 'Data dir path.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_class', 2, 'Size of stochastic vector.')
flags.DEFINE_integer('vocab_size', 10000, 'Vocabulary size.')
FLAGS = flags.FLAGS

glove_url = os.path.join(FLAGS.data_dir, 'glove.6B.100d.txt')
data_dir = os.path.join(FLAGS.data_dir, FLAGS.dataset)
vocab_url = os.path.join(data_dir, 'vocab.new')
word_dict = utils.build_vocab(vocab_url)
test_url = os.path.join(data_dir, 'test.feat')

test_target, test_set, test_count = utils.data_set(test_url)
test_batches = utils.create_batches(len(test_set), FLAGS.batch_size, shuffle=False)

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    sess = tf.compat.v1.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        x_placeholder = graph.get_operation_by_name("input").outputs[0]
        y_placeholder = graph.get_operation_by_name("output").outputs[0]
        # mask_placeholder = graph.get_operation_by_name("label_mask").outputs[0]
        batch_size_placeholder = graph.get_operation_by_name("batch_size").outputs[0]

        # Tensors we want to evaluate
        y_hat_tensor = graph.get_tensor_by_name("classifier/Softmax:0")

        test_preds = []

        for idx_batch in test_batches:
            target_batch, data_batch, count_batch, num_example = utils.fetch_data(
                test_target, test_set, test_count, idx_batch, FLAGS.n_class, FLAGS.vocab_size)
            # mask = np.array([1.0] * num_example)
            input_feed = {x_placeholder: data_batch, y_placeholder: target_batch, batch_size_placeholder: num_example}
            y_hat = sess.run(y_hat_tensor, input_feed)
            predictions = np.argmax(y_hat, axis=1)
            test_preds = np.concatenate([test_preds, predictions])

print("###################### Classification Report ######################")
print(classification_report(test_target, test_preds))
print("###################### Macro Average ######################")
print(precision_score(test_target, test_preds, average='macro'))
print(recall_score(test_target, test_preds, average='macro'))
print(f1_score(test_target, test_preds, average='macro'))
