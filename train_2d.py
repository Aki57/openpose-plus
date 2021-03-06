#!/usr/bin/env python3

import math
import multiprocessing
import os
import time
import sys

import cv2
import matplotlib
matplotlib.use('Agg')

import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt

import _pickle as cPickle

sys.path.append('.')

from train_config import config
from openpose_plus.models import model
from openpose_plus.utils import PoseInfo, get_heatmap, get_vectormap, tf_repeat, draw_results

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

tl.files.exists_or_mkdir(config.LOG.vis_path, verbose=False)  # to save visualization results
tl.files.exists_or_mkdir(config.MODEL.model_path, verbose=False)  # to save model files

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# FIXME: Don't use global variables.
# define hyper-parameters for training
node_num = config.TRAIN.node_num
batch_size = config.TRAIN.batch_size
n_step = config.TRAIN.n_step
save_interval = config.TRAIN.save_interval
weight_decay_factor = config.TRAIN.weight_decay_factor
lr_init = config.TRAIN.lr_init
lr_decay_interval = config.TRAIN.lr_decay_interval
lr_decay_factor = config.TRAIN.lr_decay_factor

# FIXME: Don't use global variables.
# define hyper-parameters for model
model_path = config.MODEL.model_path
n_pos = config.MODEL.n_pos
hin = config.MODEL.hin
win = config.MODEL.win
hout = config.MODEL.hout
wout = config.MODEL.wout


def get_pose_data_list(data_path, metas_filename, min_count, min_score):
    """
    data_path : image and anno folder name
    """
    print("[x] Get pose data from {}".format(data_path))
    data = PoseInfo(data_path, metas_filename, min_count, min_score)
    rgbs_file_list, depths_file_list, anno2ds_list = data.get_2d_data_list()
    if len(rgbs_file_list) != len(anno2ds_list) or len(depths_file_list) != len(anno2ds_list):
        raise Exception("number of images, cameras and annotations do not match")
    else:
        print("{} has {} groups of images".format(data_path, len(rgbs_file_list)))
    return rgbs_file_list, depths_file_list, anno2ds_list


def make_model(input, results, is_train=True, reuse=False):
    confs = results[:, :, :, :n_pos]
    pafs = results[:, :, :, n_pos:]

    cnn, b1_list, b2_list, base_net = model(input, n_pos, is_train, reuse)

    # define loss
    losses = []
    stage_losses = []

    for _, (l1, l2) in enumerate(zip(b1_list, b2_list)):
        loss_l1 = tf.nn.l2_loss(l1.outputs - confs)
        loss_l2 = tf.nn.l2_loss(l2.outputs - pafs)

        losses.append(tf.reduce_mean([loss_l1, loss_l2]))
        stage_losses.append(loss_l1 / batch_size)
        stage_losses.append(loss_l2 / batch_size)

    last_conf = b1_list[-1].outputs
    last_paf = b2_list[-1].outputs
    l2_loss = 0.0

    for p in tl.layers.get_variables_with_name('kernel', True, True):
        l2_loss += tf.contrib.layers.l2_regularizer(weight_decay_factor)(p)
    base_loss = tf.reduce_sum(losses) / batch_size + l2_loss

    base_net.cnn = cnn
    base_net.input = input  # base_net input
    base_net.last_conf = last_conf  # base_net output
    base_net.last_paf = last_paf  # base_net output
    base_net.confs = confs  # GT
    base_net.pafs = pafs  # GT
    base_net.stage_losses = stage_losses
    base_net.l2_loss = l2_loss
    return base_net, base_loss


def _2d_data_aug_fn(rgb_image, depth_list, ground_truth2d):
    """Data augmentation function."""
    annos2d = cPickle.loads(ground_truth2d)
    annos2d = list(annos2d)

    depth_image = sio.loadmat(depth_list)['depthim_incolor']
    depth_image = depth_image / 20.0

    ## 2d data augmentation
    # random transfrom
    M_rotate = tl.prepro.affine_rotation_matrix(angle=(-30, 30))  # original paper: -40~40
    M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.5, 0.8))  # original paper: 0.5~1.1
    M_combined = M_rotate.dot(M_zoom)
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=rgb_image.shape[0], y=rgb_image.shape[1])
    rgb_image = tl.prepro.affine_transform_cv2(rgb_image, transform_matrix)
    depth_image = tl.prepro.affine_transform_cv2(depth_image, transform_matrix, border_mode='replicate')
    annos2d = tl.prepro.affine_transform_keypoints(annos2d, transform_matrix)
    # random crop and flip
    rgb_image, annos2d, depth_image = tl.prepro.keypoint_random_flip(rgb_image, annos2d, depth_image, prob=0.5)    
    rgb_image, annos2d, depth_image = tl.prepro.keypoint_resize_random_crop(rgb_image, annos2d, depth_image, size=(hin, win)) # hao add
    # concat augmented rgb and depth
    depth_image = np.expand_dims(depth_image, axis=2)
    input_2d = np.concatenate((rgb_image, depth_image), axis=2)

    # generate 2d result maps including keypoints heatmap, pafs
    height, width, _ = input_2d.shape
    heatmap = get_heatmap(annos2d, height, width)
    vectormap = get_vectormap(annos2d, height, width)
    result2dmap = np.concatenate((heatmap, vectormap), axis=2)

    input_2d = np.array(input_2d, dtype=np.float32)
    result2dmap = np.array(result2dmap, dtype=np.float32)

    return input_2d, result2dmap


def _map_fn(rgb_list, depth_list, anno2ds):
    """TF Dataset pipeline."""
    rgb_img = tf.read_file(rgb_list)
    rgb_img = tf.image.decode_jpeg(rgb_img, channels=3)  # get RGB with 0~1
    rgb_img = tf.image.convert_image_dtype(rgb_img, dtype=tf.float32)

    # Affine transform and get paf maps
    input_2d, result2dmap = tf.py_func(_2d_data_aug_fn, [rgb_img,depth_list,anno2ds], [tf.float32, tf.float32])

    input_2d = tf.reshape(input_2d, [hin, win, 4])
    result2dmap = tf.reshape(result2dmap, [hout, wout, n_pos*3])

    input_2d = tf.image.random_brightness(input_2d, max_delta=45./255.)   # 64./255. 32./255.)  caffe -30~50
    input_2d = tf.image.random_contrast(input_2d, lower=0.5, upper=1.5)   # lower=0.2, upper=1.8)  caffe 0.3~1.5
    input_2d = tf.clip_by_value(input_2d, clip_value_min=0.0, clip_value_max=1.0)

    return input_2d, result2dmap


def single_train(training_dataset):
    ds = training_dataset.shuffle(buffer_size=4096)  # shuffle before loading images
    ds = ds.repeat(n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count() // 2)  # decouple the heavy map_fn
    ds = ds.batch(batch_size)  # TODO: consider using tf.contrib.map_and_batch
    ds = ds.prefetch(2)
    iterator = ds.make_one_shot_iterator()
    one_element = iterator.get_next()
    base_net, base_loss = make_model(*one_element, is_train=True, reuse=False)
    x_2d_ = base_net.input  # base_net input
    last_conf = base_net.last_conf  # base_net output
    last_paf = base_net.last_paf  # base_net output
    confs_ = base_net.confs  # GT
    pafs_ = base_net.pafs  # GT
    stage_losses = base_net.stage_losses
    l2_loss = base_net.l2_loss

    print('Start - n_step: {} batch_size: {} lr_init: {} lr_decay_interval: {}'.format(
        n_step, batch_size, lr_init, lr_decay_interval))

    lr_v = tf.Variable(lr_init, trainable=False, name='learning_rate')
    global_step = tf.Variable(1, trainable=False)
    train_op = tf.train.MomentumOptimizer(lr_v, 0.9).minimize(base_loss, global_step=global_step)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    # start training
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # restore pre-trained weights
        try:
            tl.files.load_and_assign_npz_dict(sess=sess, name=os.path.join(model_path, 'openposenet.npz'))
        except:
            print("no pre-trained model")

        # train until the end
        sess.run(tf.assign(lr_v, lr_init))
        while True:
            tic = time.time()
            step = sess.run(global_step)
            if step != 0 and (step % lr_decay_interval == 0):
                new_lr_decay = lr_decay_factor**(step // lr_decay_interval)
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))

            [_, _loss, _stage_losses, _l2] = sess.run([train_op, base_loss, stage_losses, l2_loss])

            # tstring = time.strftime('%d-%m %H:%M:%S', time.localtime(time.time()))
            lr = sess.run(lr_v)
            print('Total Loss at iteration {} / {} is: {} Learning rate {:10e} l2_loss {:10e} Took: {}s'.format(
                step, n_step, _loss, lr, _l2, time.time() - tic))
            for ix, ll in enumerate(_stage_losses):
                print('Network#', ix, 'For Branch', ix % 2 + 1, 'Loss:', ll)

            # save intermediate results and model
            if (step != 0) and (step % save_interval == 0):
                # save some results
                [img_out, confs_ground, pafs_ground, conf_result, paf_result] = sess.run([x_2d_, confs_, pafs_, last_conf, last_paf])
                draw_results(img_out[:,:,:,:3], confs_ground, conf_result, pafs_ground, paf_result, None, 'train_%d_' % step)
                # save model
                tl.files.save_npz_dict(base_net.all_params, os.path.join(model_path, 'openposenet' + str(step) + '.npz'), sess=sess)
                tl.files.save_npz_dict(base_net.all_params, os.path.join(model_path, 'openposenet.npz'), sess=sess)
            if step == n_step:  # training finished
                break


if __name__ == '__main__':

    if 'custom' in config.DATA.train_data:
        ## read your own images contains valid people
        ##   data/your_data
        ##           /KINECTNODE1
        ##               meta.mat
        ##               anno_01_00000118.mat
        ##               color_01_00000118.jpg
        ##               depth_01_00000118.mat
        ##           /KINECTNODE2
        ##           ...
        ## have a folder with many folders: (which is common in industry)
        root_list = tl.files.load_folder_list(path=config.DATA.data_path)
        sum_rgbs_file_list, sum_depths_file_list, sum_anno2ds_list = [], [], []
        for root in root_list:
            folder_list = tl.files.load_folder_list(path=root)
            for folder in folder_list:
                if config.DATA.image_path in folder:
                    _rgbs_file_list, _depths_file_list, _anno2ds_list = \
                        get_pose_data_list(folder, config.DATA.anno_name, 5, 0.25)
                    sum_rgbs_file_list.extend(_rgbs_file_list)
                    sum_depths_file_list.extend(_depths_file_list)
                    sum_anno2ds_list.extend(_anno2ds_list)
        print("Total number of own images found:", len(sum_rgbs_file_list))

    # define data augmentation
    def generator():
        """TF Dataset generator."""
        for _input_rgb, _input_depth, _target_anno2d in zip(sum_rgbs_file_list, sum_depths_file_list, sum_anno2ds_list):
            yield _input_rgb.encode('utf-8'), _input_depth.encode('utf-8'), cPickle.dumps(_target_anno2d)

    n_epoch = math.ceil(n_step / (len(sum_rgbs_file_list) / batch_size))
    dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.string, tf.string, tf.string))

    if config.TRAIN.train_mode == 'single':
        single_train(dataset)
    else:
        raise Exception('Unknown training mode')
