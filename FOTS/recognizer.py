import tensorflow as tf
from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('featureNum_reduceHeight', 256, '')
tf.app.flags.DEFINE_integer('num_hidden', 128, '')
tf.app.flags.DEFINE_integer('width_stride', 1, '')
FLAGS = tf.app.flags.FLAGS

class recognizer(object):
    def __init__(self, features, rnnSeqLengths, NUM_CLASSES, keepProb, weight_decay=1e-5, is_training=True):
        self.features = features
        self.rnnSeqLengths = tf.cast(tf.ceil(tf.divide(rnnSeqLengths, FLAGS.width_stride)), tf.int32)
        self.keepProb = keepProb

        self.NUM_CLASSES = NUM_CLASSES
        self.featureNum_reduceHeight = FLAGS.featureNum_reduceHeight
        self.num_hidden = FLAGS.num_hidden

        self.convLayers(weight_decay, is_training)
        self.lstmLayers(is_training)

    def loss(self, label, mask):
        # label:  sparse_placeholder
        # calout shape:  [times, b, NUM_CLASSES]
        ctc_loss = tf.nn.ctc_loss(labels=label, inputs=self.calout, sequence_length=self.rnnSeqLengths, time_major=True, 
            ignore_longer_outputs_than_inputs=True)

        cost = tf.reduce_mean(ctc_loss * mask)
        return cost

    def convLayers(self, weight_decay=1e-5, is_training=True):
        x = self.features
        print('Shape of recognizer features {}'.format(x.shape))
        # features shape: [b, fix_RoiHeight/features_stride, max_RoiWidth/features_stride, c] = [b, 8, 64, c]
        with tf.variable_scope('recg_feature', values=[self.features]):
            batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
            }
            with slim.arg_scope([slim.conv2d],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_regularizer=slim.l2_regularizer(weight_decay)):
                x = slim.conv2d(x, self.featureNum_reduceHeight, 3)
                x = slim.conv2d(x, self.featureNum_reduceHeight, 3)
                x = slim.conv2d(x, self.featureNum_reduceHeight, 3)
                x = slim.max_pool2d(x, kernel_size=[2,2], stride=[2,1], padding='SAME')
                print('+Shape of recognizer features {}'.format(x.shape))
                x = slim.conv2d(x, self.featureNum_reduceHeight, 3)
                x = slim.conv2d(x, self.featureNum_reduceHeight, 3)
                x = slim.conv2d(x, self.featureNum_reduceHeight, 3)
                x = slim.max_pool2d(x, kernel_size=[2,2], stride=[2,1], padding='SAME')
                print('++Shape of recognizer features {}'.format(x.shape))
                x = slim.conv2d(x, self.featureNum_reduceHeight, 3)
                x = slim.conv2d(x, self.featureNum_reduceHeight, 3)
                x = slim.max_pool2d(x, kernel_size=[2,2], stride=[2,1], padding='SAME')
                print('+++Shape of recognizer features {}'.format(x.shape))
                x = slim.conv2d(x, self.num_hidden, 3)
                print('++++Shape of recognizer features {}'.format(x.shape))
        self.word_vec = tf.transpose(tf.reshape(x, (-1, x.shape[2], x.shape[3])), (1, 0, 2))
        print('++++Shape of word_vec {}'.format(self.word_vec.shape))
        # word_vec shape: [times, b, num_hidden] = [32, b, num_hidden]


    def lstmLayers(self, is_training=True):
        with tf.variable_scope('lstmLayers') as scope:
            lstmFwCell_l1 = tf.contrib.rnn.LSTMCell(self.num_hidden, forget_bias=1.0, state_is_tuple=True)
            lstmFwCell_l2 = tf.contrib.rnn.LSTMCell(self.num_hidden, forget_bias=1.0, state_is_tuple=True)
            lstmFwCell = tf.contrib.rnn.MultiRNNCell([lstmFwCell_l1, lstmFwCell_l2], state_is_tuple=True) 
            lstmFwCell = tf.contrib.rnn.DropoutWrapper(lstmFwCell, input_keep_prob=self.keepProb, output_keep_prob=self.keepProb)
            # lstmFwCell_init_state = lstmFwCell.zero_state(self.rnnBatch, dtype=tf.float32)
            lstmBwCell_l1 = tf.contrib.rnn.LSTMCell(self.num_hidden, forget_bias=1.0, state_is_tuple=True)
            lstmBwCell_l2 = tf.contrib.rnn.LSTMCell(self.num_hidden, forget_bias=1.0, state_is_tuple=True)
            lstmBwCell = tf.contrib.rnn.MultiRNNCell([lstmBwCell_l1, lstmBwCell_l2], state_is_tuple=True)
            lstmBwCell = tf.contrib.rnn.DropoutWrapper(lstmBwCell, input_keep_prob=self.keepProb, output_keep_prob=self.keepProb)
            # lstmBwCell_init_state = lstmBwCell.zero_state(self.rnnBatch, dtype=tf.float32)

            lstmout, _ = tf.nn.bidirectional_dynamic_rnn(
                          lstmFwCell,
                          lstmBwCell,
                          self.word_vec,
                          sequence_length=self.rnnSeqLengths,
                          initial_state_fw=None,
                          initial_state_bw=None,
                          dtype=tf.float32,
                          parallel_iterations=None,
                          swap_memory=False,
                          time_major=True,
                          scope=''
                      )
            lstmout = tf.concat(lstmout, 2)
            # lstmout shape:  [times, b, 2 * num_hidden]
            self.logits = slim.fully_connected(lstmout, self.NUM_CLASSES, activation_fn=None, scope='fc')
            self.calout = self.logits
            self.logits = tf.transpose(self.logits, (1, 0, 2))
            # logits shape:  [b, times, NUM_CLASSES]
            print('++++Shape of logits {}'.format(self.logits.shape))
            # logits shape:  [b, times, NUM_CLASSES]
            self.conf = tf.nn.softmax(self.logits)
            # conf shape:  [b, times, NUM_CLASSES]
            self.ans = tf.argmax(self.logits, axis=2)
            # ans shape:  [b, times]

