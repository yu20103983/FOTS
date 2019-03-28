import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('input_size', 224, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 32, '')
tf.app.flags.DEFINE_integer('num_readers', 8, '')
tf.app.flags.DEFINE_float('learning_rate', 0.001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('train_gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'dataFolder/FOST_textOnPlate', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer('test_steps', -1, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')

from FOTS.dataset import dataReader
from FOTS.fots_trainModel import FOTS_trainModel
from FOTS.fots_testModel import FOTS_testModel

FLAGS = tf.app.flags.FLAGS

def sparse_tuple_from_label(sequences):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(0,len(seq),1)))
        values.extend(seq)

    indices = np.asarray(indices)
    values = np.asarray(values)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1])
    return indices, values, shape

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

def build_model(opt, reuse_variables, scope=None):
    input_images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size_per_gpu, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[FLAGS.batch_size_per_gpu, None, None, 1], name='input_score_maps')
    input_geo_maps = tf.placeholder(tf.float32, shape=[FLAGS.batch_size_per_gpu, None, None, 5], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[FLAGS.batch_size_per_gpu, None, None, 1], name='input_training_masks')
    input_brboxes = []
    for i in range(FLAGS.batch_size_per_gpu):
        outBoxes = tf.placeholder(tf.int32, shape=[None, 4], name='input_outBoxes')
        cropBoxes = tf.placeholder(tf.int32, shape=[None, 4], name='input_cropBoxes')
        angles = tf.placeholder(tf.float32, shape=[None,], name='input_angles')
        input_brboxes.append((outBoxes, cropBoxes, angles))

    input_btags = tf.sparse_placeholder(tf.int32, name='input_btags')
    input_recg_masks = tf.placeholder(tf.float32, name='input_recg_masks')
    fots = FOTS_trainModel(input_images, input_brboxes, reuse_variables)
    total_loss, model_loss, detector_loss, recognizer_loss = fots.total_loss(input_score_maps, input_geo_maps, input_training_masks, input_btags, input_recg_masks)
    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
    grads = opt.compute_gradients(total_loss)
    return input_images, input_score_maps, input_geo_maps, input_training_masks, input_brboxes, input_btags, input_recg_masks, total_loss, model_loss, detector_loss, recognizer_loss, batch_norm_updates_op, grads


def main(argv=None):
    if len(argv) >= 2:
        gpu_list = argv[1]
    else:
        gpu_list = FLAGS.train_gpu_list
    gpus = [int(i) for i in gpu_list.split(',')] if len(gpu_list) > 0 else None
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    reuse_variables = None
    models = []
    if gpus:
        for i, gpu_id in enumerate(gpus):
            with tf.device('/device:GPU:%d' % gpu_id):
                print 'device : /gpu:%d' % gpu_id
                with tf.name_scope('model_%d' % gpu_id) as scope:
                    models.append(build_model(opt, reuse_variables, scope))
                    reuse_variables = True
    else:
        models.append(build_model(opt, reuse_variables))

    tower_total_loss, tower_model_loss, tower_detector_loss, tower_recognizer_loss, tower_batch_norm_updates_op, tower_grads = zip(*models)[-6:]

    grads = average_gradients(tower_grads)
    batch_norm_updates_op = tf.group(*tower_batch_norm_updates_op)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    total_loss = tf.reduce_mean(tf.concat(tower_total_loss, 0))
    model_loss = tf.reduce_mean(tf.concat(tower_model_loss, 0))
    detector_loss = tf.reduce_mean(tf.concat(tower_detector_loss, 0))
    recognizer_loss = tf.reduce_mean(tf.concat(tower_recognizer_loss, 0))

    tf.summary.scalar('detector_loss', detector_loss)
    tf.summary.scalar('recognizer_loss', recognizer_loss)
    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('total_loss', total_loss)


    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.pretrained_model_path)
        pretrained_model_path = os.path.join(FLAGS.pretrained_model_path,
                                             os.path.basename(ckpt_state.model_checkpoint_path))
        variable_restore_op = slim.assign_from_checkpoint_fn(pretrained_model_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = dataReader.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * (len(gpus) if gpus else 1))

        fots_testModel = FOTS_testModel(reuse_variables=True)
        start = time.time()
        for step in range(FLAGS.max_steps):
            d_images, _, d_score_maps, d_geo_maps, d_training_masks, d_brboxes, d_btags, d_bRecgTags = next(data_generator)
            inp_dict = {}
            count = 0
            for m in models:
                input_images, input_score_maps, input_geo_maps, input_training_masks, input_brboxes, input_btags, input_brecg_masks = m[:7]
                inp_dict[input_images] = d_images[count:count + FLAGS.batch_size_per_gpu]
                inp_dict[input_score_maps] = d_score_maps[count:count + FLAGS.batch_size_per_gpu]
                inp_dict[input_geo_maps] = d_geo_maps[count:count + FLAGS.batch_size_per_gpu]
                inp_dict[input_training_masks] = d_training_masks[count:count + FLAGS.batch_size_per_gpu]
                for j in range(FLAGS.batch_size_per_gpu):
                    inp_dict[input_brboxes[j][0]] = d_brboxes[count + j][0]  # outBoxs
                    inp_dict[input_brboxes[j][1]] = d_brboxes[count + j][1]  # cropBoxs
                    inp_dict[input_brboxes[j][2]] = d_brboxes[count + j][2]  # angles
                cur_d_btags = d_btags[count:count + FLAGS.batch_size_per_gpu]
                cur_d_btags = [j for i in cur_d_btags for j in i]

                cur_d_bRecgTags = d_bRecgTags[count:count + FLAGS.batch_size_per_gpu]
                cur_d_bRecgTags = np.array([j for i in cur_d_bRecgTags for j in i], np.float32)

                cur_d_btags = sparse_tuple_from_label(cur_d_btags)

                inp_dict[input_btags] = cur_d_btags
                inp_dict[input_brecg_masks] = cur_d_bRecgTags
                count += FLAGS.batch_size_per_gpu

            dl, rl, ml, tl, _ = sess.run([detector_loss, recognizer_loss, model_loss, total_loss, train_op], feed_dict=inp_dict)
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start)/10
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * (len(gpus) if gpus else 1))/(time.time() - start)
                start = time.time()
                print('Step {:06d}, total loss {:.4f}, detector loss {:.4f}, recognizer loss {:.4f}, model loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                    step, tl, dl, rl, ml, avg_time_per_step, avg_examples_per_second))

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_path, 'model.ckpt'), global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict= inp_dict)
                summary_writer.add_summary(summary_str, global_step=step)

            if step % FLAGS.test_steps == 0 and FLAGS.test_steps > 0:
                fots_testModel.detectRecg(d_images, sess=sess)


if __name__ == '__main__':
    tf.app.run()
