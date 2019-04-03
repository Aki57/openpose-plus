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
from openpose_plus.utils import PoseInfo, read_depth, get_heatmap, get_vectormap, tf_repeat, draw_results

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

tl.files.exists_or_mkdir(config.LOG.vis_path, verbose=False)  # to save visualization results
tl.files.exists_or_mkdir(config.MODEL.model_path, verbose=False)  # to save model files

# FIXME: Don't use global variables.
# define hyper-parameters for training
batch_size = config.TRAIN.batch_size
n_epoch = config.TRAIN.n_epoch
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

    rgb_image = 255 - rgb_image if np.random.uniform() < 0.25 else rgb_image

    depth_image = read_depth(depth_list.decode())
    depth_image = depth_image / np.random.uniform(20, 40)
    depth_image = tl.prepro.drop(depth_image, keep=np.random.uniform())

    ## 2d data augmentation
    # random transfrom
    M_rotate = tl.prepro.affine_rotation_matrix(angle=(-40, 40))  # original paper: -40~40 -> -30~30
    M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.5, 1.1))  # original paper: 0.5~1.1 -> 0.5~0.8
    M_combined = M_rotate.dot(M_zoom)
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=rgb_image.shape[1], y=rgb_image.shape[0])
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

    input_2d = tf.image.random_brightness(input_2d, max_delta=32./255.)   # 64./255. 32./255.)  caffe -30~50
    input_2d = tf.image.random_contrast(input_2d, lower=0.3, upper=1.5)   # lower=0.2, upper=1.8)  caffe 0.3~1.5
    input_2d = tf.clip_by_value(input_2d, clip_value_min=0.0, clip_value_max=1.0)

    return input_2d, result2dmap


def train(training_dataset, epoch, n_step):
    ds = training_dataset.shuffle(buffer_size=4096)  # shuffle before loading images
    ds = ds.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count() // 2)  # decouple the heavy map_fn
    ds = ds.batch(batch_size)
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

    new_lr_decay = lr_decay_factor**((epoch-1)*n_step // lr_decay_interval)
    print('Start - epoch: {} n_step: {} batch_size: {} lr_init: {} lr_decay_interval: {}'.format(
        epoch, n_step, batch_size, lr_init*new_lr_decay, lr_decay_interval))

    lr_v = tf.Variable(lr_init * new_lr_decay, trainable=False, name='learning_rate')
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
            if step != 0 and (((epoch-1)*n_step + step) % lr_decay_interval == 0):
                new_lr_decay = lr_decay_factor**(((epoch-1)*n_step + step) // lr_decay_interval)
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                print('lr decay to {}'.format(lr_init*new_lr_decay))

            [_, _loss, _stage_losses, _l2] = sess.run([train_op, base_loss, stage_losses, l2_loss])

            # tstring = time.strftime('%d-%m %H:%M:%S', time.localtime(time.time()))
            lr = sess.run(lr_v)
            print('Training Loss at iteration {} / {} is: {} Learning rate {:10e} l2_loss {:10e} Took: {}s'.format(
                step, n_step, _loss, lr, _l2, time.time() - tic))
            for ix, ll in enumerate(_stage_losses):
                print('Network#', ix, 'For Branch', ix % 2 + 1, 'Loss:', ll)

            # save intermediate results and model
            if (step != 0) and (step % save_interval == 0):
                # save some results
                [img_out, confs_ground, pafs_ground, conf_result, paf_result] = sess.run([x_2d_, confs_, pafs_, last_conf, last_paf])
                draw_results(img_out[:,:,:,:3], confs_ground, conf_result, pafs_ground, paf_result, None, 'train_%d_' % step)
                # save model
                tl.files.save_npz_dict(base_net.all_params, os.path.join(model_path, 'openposenet-'+str(epoch)+'-'+str(step)+'.npz'), sess=sess)
                tl.files.save_npz_dict(base_net.all_params, os.path.join(model_path, 'openposenet.npz'), sess=sess)
            # training finished
            if step == n_step:
                tl.files.save_npz_dict(base_net.all_params, os.path.join(model_path, 'openposenet-'+str(epoch)+'.npz'), sess=sess)
                tl.files.save_npz_dict(base_net.all_params, os.path.join(model_path, 'openposenet.npz'), sess=sess)
                break


def evaluate(evaluating_dataset, epoch, n_step):
    ds = evaluating_dataset.shuffle(buffer_size=4096)  # shuffle before loading images
    ds = ds.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count() // 2)  # decouple the heavy map_fn
    ds = ds.batch(batch_size)
    ds = ds.prefetch(2)
    iterator = ds.make_one_shot_iterator()
    one_element = iterator.get_next()
    base_net, base_loss = make_model(*one_element, is_train=True, reuse=False)
    l2_loss = base_net.l2_loss

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    # start evaluating
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # restore pre-trained weights
        try:
            tl.files.load_and_assign_npz_dict(sess=sess, name=os.path.join(model_path, 'openposenet.npz'))
        except:
            print("no pre-trained model")

        # evaluate all test files
        step = 0
        sum_loss = 0.0
        invalid_count = 0
        while True:
            tic = time.time()
            [_loss, _l2] = sess.run([base_loss, l2_loss])

            step += 1
            if _loss == _l2:
                invalid_count += 1
            else:
                sum_loss += _loss

            print('Validation loss at iteration {} / {} is: {} Took: {}s'.format(step, n_step, _loss, time.time() - tic))
            if step == n_step:
                break

        # evaluating finished
        avg_loss = sum_loss / (n_step-invalid_count)
        print('Total validation average loss at epoch {} is: {} Took: {}s'.format(epoch, avg_loss, time.time() - tic))


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

    from sklearn.model_selection import train_test_split
    train_rgbs_list, eval_rgbs_list, \
    train_depths_list, eval_depths_list, \
    train_anno2ds_list, eval_anno2ds_list = train_test_split(
        sum_rgbs_file_list, sum_depths_file_list, sum_anno2ds_list, test_size=0.2, random_state=42)
    print("{} images for training, {} images for evalutation.".format(len(train_rgbs_list), len(eval_rgbs_list)))

    # define data augmentation
    def train_ds_generator():
        """TF Dataset generator."""
        for _input_rgb, _input_depth, _target_anno2d in zip(train_rgbs_list, train_depths_list, train_anno2ds_list):
            yield _input_rgb.encode('utf-8'), _input_depth.encode('utf-8'), cPickle.dumps(_target_anno2d)
    train_ds = tf.data.Dataset().from_generator(train_ds_generator, output_types=(tf.string, tf.string, tf.string))

    def eval_ds_generator():
        """TF Dataset generator."""
        for _input_rgb, _input_depth, _target_anno2d in zip(eval_rgbs_list, eval_depths_list, eval_anno2ds_list):
            yield _input_rgb.encode('utf-8'), _input_depth.encode('utf-8'), cPickle.dumps(_target_anno2d)
    eval_ds = tf.data.Dataset().from_generator(eval_ds_generator, output_types=(tf.string, tf.string, tf.string))

    for epoch in range(1, n_epoch+1):
        tf.reset_default_graph()
        train(train_ds, epoch, (int)(len(train_rgbs_list)//batch_size))
        tf.reset_default_graph()
        evaluate(eval_ds, epoch, (int)(len(eval_rgbs_list)//batch_size))