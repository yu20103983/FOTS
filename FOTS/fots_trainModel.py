import tensorflow as tf
import numpy as np

import detector
import recognizer
import RoiRotate
import sharedConv
from icdar import NUM_CLASSES

tf.app.flags.DEFINE_float('keepProb', 0.8, '')
tf.app.flags.DEFINE_float('alpha', 1., '')
tf.app.flags.DEFINE_float('beta', 1., '')
FLAGS = tf.app.flags.FLAGS

class FOTS_trainModel():
    def __init__(self, images, brboxes, reuse_variables=None):
        self.reuse_variables = reuse_variables
        with tf.variable_scope(tf.get_variable_scope(), reuse=self.reuse_variables):
            self.sharedFeatures = sharedConv.model(images, is_training=True)
            self.det = detector.detector(self.sharedFeatures)
            self.rois, self.ws = RoiRotate.RoiRotate(self.sharedFeatures, FLAGS.features_stride)(brboxes)
            self.rg = recognizer.recognizer(self.rois, self.ws, NUM_CLASSES, FLAGS.keepProb, is_training=True)

        # add summary
        if self.reuse_variables is None:
            org_rois, org_ws = RoiRotate.RoiRotate(images, 1)(brboxes, expand_w=60)
            # org_rois shape [b, 8, 64, 3]
            tf.summary.image('input', images)
            tf.summary.image('score_map_pred', self.det.F_score * 255)
            tf.summary.image('geo_map_0_pred', self.det.F_geometry[:, :, :, 0:1])
            tf.summary.image('org_rois', org_rois, max_outputs=12)

    def model_loss(self, score_maps, geo_maps, training_masks, btags, recg_masks):
        self.detector_loss = self.det.loss(score_maps, geo_maps, training_masks)
        self.recognizer_loss = self.rg.loss(btags, recg_masks)
        self.model_loss = FLAGS.beta * self.detector_loss + FLAGS.alpha * self.recognizer_loss

    def total_loss(self, score_maps, geo_maps, training_masks, btags, recg_masks):
        self.model_loss(score_maps, geo_maps, training_masks, btags, recg_masks)
        self.total_loss = tf.add_n([self.model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        if self.reuse_variables is None:
            tf.summary.image('score_map', score_maps)
            tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
            tf.summary.image('training_masks', training_masks)
        return self.total_loss, self.model_loss, self.detector_loss, self.recognizer_loss