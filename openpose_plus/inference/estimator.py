import logging
import time

import tensorflow as tf
import tensorlayer as tl

from .post_process import PostProcessor, detect_scorevol

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class TfPoseEstimator:

    def __init__(self, graph_path, model_func, target_size=(368, 368, 3), data_format='channels_last'):
        height, width, _ = target_size
        f_height, f_width = (height / 8, width / 8)
        self.post_processor = PostProcessor((height, width), (f_height, f_width), data_format)

        self.tensor_input2d, self.tensor_heatmap, self.tensor_paf = model_func(target_size, data_format)
        self._warm_up(graph_path)

    def _warm_up(self, graph_path):
        self.persistent_sess = tf.InteractiveSession()
        self.persistent_sess.run(tf.global_variables_initializer())
        tl.files.load_and_assign_npz_dict(graph_path, self.persistent_sess)

    def __del__(self):
        self.persistent_sess.close()

    def inference(self, input_2d):
        heatmap, pafmap = self.persistent_sess.run(
            [self.tensor_heatmap, self.tensor_paf], feed_dict={self.tensor_input2d: [input_2d],})

        humans, heatmap_up, pafmap_up = self.post_processor(heatmap[0], pafmap[0])
        return humans, heatmap_up, pafmap_up


class Pose3DEstimator:

    def __init__(self, graph_path, model_func, target_size=(64, 64, 64), use_slim=False, data_format='channels_last'):
        self.tensor_input3d, self.tensor_voxel = model_func(target_size, use_slim, data_format)
        self._warm_up(graph_path)

    def _warm_up(self, graph_path):
        self.persistent_sess = tf.InteractiveSession()
        self.persistent_sess.run(tf.global_variables_initializer())
        tl.files.load_and_assign_npz_dict(graph_path, self.persistent_sess)

    def __del__(self):
        self.persistent_sess.close()

    def inference(self, input_3d):
        out_voxel = self.persistent_sess.run(self.tensor_voxel, feed_dict={self.tensor_input3d: input_3d,})

        coords_xyz, coords_conf = detect_scorevol(out_voxel)
        return coords_xyz, coords_conf

    def regression(self, input_3d):
        _, xdim, ydim, zdim, n_chan = input_3d.shape
        grid = tf.meshgrid(tf.range(0.0, xdim), tf.range(0.0, ydim), tf.range(0.0, zdim), indexing='ij')
        grid = tf.tile(tf.expand_dims(grid,-1), [1,1,1,1,n_chan-1])

        self.tensor_voxel = tf.squeeze(self.tensor_voxel)
        self.tensor_voxel = tf.exp(self.tensor_voxel - tf.reduce_max(self.tensor_voxel,[0,1,2]))
        self.tensor_voxel = self.tensor_voxel / tf.reduce_sum(self.tensor_voxel, [0,1,2])
        tensor_pred = tf.transpose(tf.reduce_sum(self.tensor_voxel * grid, [1,2,3]))

        out_pred = self.persistent_sess.run(tensor_pred, feed_dict={self.tensor_input3d: input_3d,})
        return out_pred