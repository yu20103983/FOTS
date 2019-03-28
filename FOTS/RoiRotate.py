import tensorflow as tf 
import numpy as np

tf.app.flags.DEFINE_integer('virtule_RoiHeight', 32, '')
tf.app.flags.DEFINE_integer('virtule_MaxRoiWidth', 256, '')
FLAGS = tf.app.flags.FLAGS

class RoiRotate(object):
	def __init__(self, features, features_stride):
		self.features = features
		self.features_stride = features_stride

		self.max_RoiWidth = FLAGS.virtule_MaxRoiWidth / features_stride
		self.fix_RoiHeight = FLAGS.virtule_RoiHeight / features_stride
		self.ratio = float(self.fix_RoiHeight) / self.max_RoiWidth

	def scanFunc(self, state, b_input):
		ifeatures, outBox, cropBox, angle = b_input
		cropFeatures = tf.image.crop_to_bounding_box(ifeatures, outBox[1], outBox[0], outBox[3], outBox[2])
		rotateCropedFeatures = tf.contrib.image.rotate(cropFeatures, angle)

		textImgFeatures = tf.image.crop_to_bounding_box(rotateCropedFeatures, cropBox[1], cropBox[0], cropBox[3], cropBox[2])

		# resize keep ratio 	
		w = tf.cast(tf.ceil(tf.multiply(tf.divide(self.fix_RoiHeight, cropBox[3]), tf.cast(cropBox[2], tf.float64))), tf.int32)
		resize_textImgFeatures = tf.image.resize_images(textImgFeatures, (self.fix_RoiHeight, w), 1)
		w = tf.minimum(w, self.max_RoiWidth)
		pad_or_crop_textImgFeatures = tf.image.crop_to_bounding_box(resize_textImgFeatures, 0, 0, self.fix_RoiHeight, w)
		# pad
		pad_or_crop_textImgFeatures = tf.image.pad_to_bounding_box(pad_or_crop_textImgFeatures, 0, 0, self.fix_RoiHeight, self.max_RoiWidth)
		
		return [pad_or_crop_textImgFeatures, w]


	def __call__(self, brboxes, expand_w=20):
		paddings = tf.constant([[0, 0],[expand_w, expand_w], [expand_w, expand_w], [0, 0]])
		features_pad = tf.pad(self.features, paddings, "CONSTANT")
		features_pad = tf.expand_dims(features_pad, axis=1)
		# features_pad shape: [b, 1, h, w, c]
		nums = features_pad.shape[0]
		channels = features_pad.shape[-1]

		btextImgFeatures = []
		ws = []

		for b, rboxes in enumerate(brboxes):
			outBoxes, cropBoxes, angles = rboxes
			# outBoxes = tf.cast(tf.ceil(tf.divide(outBoxes, self.features_stride)), tf.int32)  # float div
			# cropBoxes = tf.cast(tf.ceil(tf.divide(cropBoxes, self.features_stride)), tf.int32) # float div

			outBoxes = tf.div(outBoxes, self.features_stride)  # int div
			cropBoxes = tf.div(cropBoxes, self.features_stride) # int div

			outBoxes_xy = outBoxes[:, :2]
			outBoxes_xy =  tf.add(outBoxes_xy, expand_w)
			outBoxes = tf.concat([outBoxes_xy, outBoxes[:, 2:]], axis=1)

			# len_crop = outBoxes.shape[0]  # error tf.stack cannot convert an unknown Dimension to a tensor: ?
			len_crop = tf.shape(outBoxes)[0]
			ifeatures_pad = features_pad[b]
			# ifeatures_tile = tf.tile(ifeatures_pad, tf.stack([len_crop, 1, 1, 1]))
			ifeatures_tile = tf.tile(ifeatures_pad, [len_crop, 1, 1, 1])

			textImgFeatures = tf.scan(self.scanFunc, [ifeatures_tile, outBoxes, cropBoxes, angles], [np.zeros((self.fix_RoiHeight, self.max_RoiWidth, channels), np.float32), np.array(0, np.int32)])
			btextImgFeatures.append(textImgFeatures[0])
			ws.append(textImgFeatures[1])

		btextImgFeatures = tf.concat(btextImgFeatures, axis=0)
		ws = tf.concat(ws, axis=0)

		return btextImgFeatures, ws


