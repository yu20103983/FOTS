import tensorflow as tf
import numpy as np
import cv2

import detector
import recognizer
import RoiRotate
import sharedConv
from icdar import NUM_CLASSES, restore_roiRotatePara, restore_rectangle, decode_maps
import locality_aware_nms as nms_locality

tf.app.flags.DEFINE_integer('test_input_size', 224, '')
tf.app.flags.DEFINE_integer('test_batch', 8, '')
tf.app.flags.DEFINE_integer('min_scale', 8, '')
tf.app.flags.DEFINE_integer('RoiHeight', 8, '')
tf.app.flags.DEFINE_integer('MaxRoiWidth', 64, '')
tf.app.flags.DEFINE_integer('sharedFeatureChannel', 32, '')

FLAGS = tf.app.flags.FLAGS

class FOTS_testModel():
    def __init__(self, model_path=None, reuse_variables=None):
        self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        self.input_rois = tf.placeholder(tf.float32, shape=[None, FLAGS.RoiHeight, FLAGS.MaxRoiWidth, FLAGS.sharedFeatureChannel], name='input_rois')
        self.input_ws = tf.placeholder(tf.float32, shape=[None,], name='input_ws')

        self.NUM_CLASSES = NUM_CLASSES
        self.decode_maps = decode_maps

        self.reuse_variables = reuse_variables
        with tf.variable_scope(tf.get_variable_scope(), reuse=self.reuse_variables):
            self.sharedFeatures = sharedConv.model(self.input_images, is_training=False)
            self.det = detector.detector(self.sharedFeatures)
            self.f_score = self.det.F_score
            self.f_geometry = self.det.F_geometry
            self.rg = recognizer.recognizer(self.input_rois, self.input_ws, self.NUM_CLASSES, 1., is_training=False)
            self.conf = self.rg.conf
            self.ans = self.rg.ans
            self.sess = None

        if model_path:
            self.checkpoint_path = model_path
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            self.variable_averages = tf.train.ExponentialMovingAverage(0.997, self.global_step)
            self.saver = tf.train.Saver(self.variable_averages.variables_to_restore())
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.05)))
            print('Restore from {}'.format(self.checkpoint_path))
            self.saver.restore(self.sess, self.checkpoint_path)

    def resize_image(self, im, max_side_len=224):
        h, w = im.shape[:2]
        max_h_w_i = np.max([h, w])
        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
        im_padded[:h, :w, :] = im.copy()
        im_ = im_padded
        # resize the image to input size
        new_h, new_w = im.shape[:2]
        resize_h = max_side_len
        resize_w = max_side_len
        im_ = cv2.resize(im_, dsize=(resize_w, resize_h))
        return im_, (resize_h / float(max_h_w_i), resize_w / float(max_h_w_i))

    def nms_boxBuild(self, score_map, geo_map, timer, ratio, score_map_thresh=0.5, box_thresh=0.1, nms_thres=0.2):
        '''
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param timer:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        '''
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, :]
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # print geo_map[np.where(score_map > score_map_thresh)][:, 4]
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        # start = time.time()
        text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
        # boxes = np.concatenate([boxes, _boxes], axis=0)

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        if len(boxes)>0:
            boxes = boxes[boxes[:, 8] > box_thresh]

        return boxes, timer

    def sort_poly(self, p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def ctc_label(self, p):
        ret = []
        p1 = [self.NUM_CLASSES - 1] + p
        for i in range(len(p)):
            c1 = p1[i]
            c2 = p1[i+1]
            if c2 == self.NUM_CLASSES - 1 or c2 == c1:
                continue
            ret.append(c2)
        return ret


    def ctc_label_confidence(self, conf, p):
        l = []
        conf = []
        p = p + [self.NUM_CLASSES - 1]
        last = self.NUM_CLASSES - 1
        ctp = 0.

        for i, v in enumerate(p):
            if v != last and last != self.NUM_CLASSES - 1:
                l.append(last)
                conf.append(ctp)
                ctp = 0.
            if v != self.NUM_CLASSES - 1:
                if conf[i][v] > ctp:
                    ctp = conf[i][v]
            last = v

        return conf, l

    def forward_detect(self, img):
        img = img if (isinstance(img, list) or img.ndim == 4) else [img]
        ratio1 = []
        size = []
        ratio2 = []
        iim_resizeds = []
        for iimg in img:
            size.append(iimg.shape[:2])
            iim_resized, iratio = self.resize_image(iimg, FLAGS.test_input_size)
            ratio1.append(iratio)
            h, w, _ = iim_resized.shape
            ratio2.append(h / w)
            iim_resizeds.append(iim_resized)
        iim_resizeds = np.array(iim_resizeds, np.float32)
        iim_resizedBatchs = np.array_split(iim_resizeds, int((len(iim_resizeds) + FLAGS.test_batch) / FLAGS.test_batch))
        scores = []
        geometrys = []
        sharedFeatures = []
        for iim_resizedBatch in iim_resizedBatchs:
            isharedFeatures, iscore, igeometry = self.sess.run([self.sharedFeatures, self.f_score, self.f_geometry], feed_dict={self.input_images: iim_resizedBatch})
            sharedFeatures.append(isharedFeatures)
            scores.append(iscore)
            geometrys.append(igeometry)
        sharedFeatures = np.concatenate(sharedFeatures, axis=0)
        scores = np.concatenate(scores, axis=0)
        geometrys = np.concatenate(geometrys, axis=0)
        return sharedFeatures, zip(size, ratio1, scores, geometrys, ratio2)

    def forward_recg(self, rois, ws):
        roiBatchs = np.array_split(rois, int((len(rois) + FLAGS.test_batch) / FLAGS.test_batch))
        wBatchs = np.array_split(ws, int((len(ws) + FLAGS.test_batch) / FLAGS.test_batch))
        confs = []
        anss = []
        for roiBatch, wBatch in zip(roiBatchs, wBatchs):
            conf, ans = self.sess.run([self.conf, self.ans], feed_dict={self.input_rois: roiBatch, self.input_ws: wBatch})
            confs.append(conf)
            anss.append(ans)
        confs = np.concatenate(confs, axis=0)
        anss = np.concatenate(anss, axis=0)
        return zip(confs, anss)

    def detect(self, im, sess=None, getFeatures = False):
        if sess:
            self.sess = sess
        if len(im) == 0:
            return np.array([], np.float32) if getFeatures is False else np.array([], np.float32), np.array([], np.float32) 
        preD = self.forward_detect(im)
        res = []
        sharedFeatures = preD[0]
        for size, ratio1, score, geometry, ratio2 in preD[1]:
            boxes, _ = self.nms_boxBuild(score_map=np.expand_dims(score, 0), geo_map=np.expand_dims(geometry, 0), timer=None, ratio=ratio2)
            ratio_h, ratio_w = ratio1
            h, w = size
            if len(boxes)>0:
                boxes[:,:8:2] /= ratio_w
                boxes[:,1:8:2] /= ratio_h
                if not getFeatures:
                    boxes[:,:8:2] = np.clip(boxes[:,:8:2], 0, w - 1)
                    boxes[:,1:8:2] = np.clip(boxes[:,1:8:2], 0, h - 1)
            result = []
            if len(boxes)>0:
                for box in boxes:
                    box_ =  box[:8].reshape((4, 2))
                    if np.linalg.norm(box_[0] - box_[1]) < FLAGS.min_scale or np.linalg.norm(box_[3]-box_[0]) < FLAGS.min_scale:
                        continue
                    result.append(box)
            res.append(np.array(result, np.float32))

        return (res if isinstance(im, list) or im.ndim == 4 else res[0]) if getFeatures is False else res, sharedFeatures 

    def detectRecg(self, im, sess=None):
        if sess:
            self.sess = sess
        brboxes, bsharedFeatures = self.detect(im, getFeatures = True)
        out_res = [[[1., brboxes[i][j], 0., ''] for j in range(len(brboxes[i]))] for i in range(len(brboxes))]
        box_index = []
        brotateParas = []
        filter_bsharedFeatures = []
        for i, rboxes, sharedFeatures in zip(range(len(brboxes)), brboxes, bsharedFeatures):
            rotateParas = []
            for j, rbox in enumerate(rboxes):
                para = restore_roiRotatePara(rbox)
                if para and min(para[1][2:]) > FLAGS.min_scale:
                    rotateParas.append(para)
                    box_index.append((i, j))

            if len(rotateParas) > 0:
                rotateParas = map(lambda x: np.array(x), zip(*rotateParas))
                filter_bsharedFeatures.append(sharedFeatures)
                brotateParas.append(rotateParas)
        if len(brotateParas) == 0:
            return out_res
        filter_bsharedFeatures = np.array(filter_bsharedFeatures, np.float32)

        rois_op, ws_op = RoiRotate.RoiRotate(filter_bsharedFeatures, FLAGS.features_stride)(brotateParas)

        rois, ws = self.sess.run([rois_op, ws_op])
        preD = self.forward_recg(ros, ws)

        for ij, conf_ans in zip(box_index, preD):
            conf, ans = conf_ans
            img_index, box_on_img_index = ij
            conf, ans = self.ctc_label_confidence(conf, ans)
            if len(conf) == 0:
                continue
            # ret=''.encode('utf-8')
            ret = ''.join([self.decode_maps[li] for li in ans])
            min_conf = np.min(conf)
            out_res[img_index][box_on_img_index][2] = min_conf
            out_res[img_index][box_on_img_index][3] = ret
        return out_res








