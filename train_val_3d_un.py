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
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt

import _pickle as cPickle

sys.path.append('.')

from train_config import config
from openpose_plus.models import model
from openpose_plus.utils import PoseInfo, read_depth, aug_depth, create_voxelgrid, get_3d_heatmap, get_kp_heatmap, keypoint_flip, keypoints_affine

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
xdim = config.MODEL.xdim
ydim = config.MODEL.ydim
zdim = config.MODEL.zdim
sigma = config.MODEL.sigma
b_slim = config.MODEL.use_slim


def get_pose_data_list(data_path, metas_filename, min_count, min_score):
    """
    data_path : image and anno folder name
    """
    print("[x] Get pose data from {}".format(data_path))
    data = PoseInfo(data_path, metas_filename, min_count, min_score)
    cams_list, depths_file_list, anno2ds_list, anno3ds_list = data.get_3d_data_list()
    if len(depths_file_list) != len(anno2ds_list) or len(anno3ds_list) != len(cams_list):
        raise Exception("number of images, cameras and annotations do not match")
    else:
        print("{} has {} groups of images".format(data_path, len(depths_file_list)))
    return depths_file_list, cams_list, anno2ds_list, anno3ds_list


def make_model(input, output, mask, reuse=False, use_slim=False):
    input = tf.reshape(input, [batch_size, xdim, ydim, zdim, n_pos+1])
    output = tf.reshape(output, [batch_size, xdim, ydim, zdim, n_pos])
    mask = tf.reshape(mask, [batch_size, xdim, ydim, zdim, n_pos])

    voxel_list, head_net = model(input, n_pos, reuse, use_slim)

    # define loss
    stage_losses = []

    for _, voxel in enumerate(voxel_list):
        loss = tf.nn.l2_loss((voxel.outputs - output) * mask)
        stage_losses.append(loss / batch_size)

    l2_loss = 0.0
    for p in tl.layers.get_variables_with_name('W_', True, True):
        l2_loss += tf.contrib.layers.l2_regularizer(weight_decay_factor)(p)
    head_loss = tf.reduce_sum(stage_losses) + l2_loss

    head_net.input = input  # base_net input
    head_net.last_voxel = head_net.outputs  # base_net output
    head_net.mask = mask
    head_net.voxels = output  # GT
    head_net.stage_losses = stage_losses
    head_net.l2_loss = l2_loss
    return head_net, head_loss


def _3d_data_aug_fn(depth_list, cam, ground_truth2d, ground_truth3d):
    """Data augmentation function."""
    # Augmentation of depth image
    dep_img = read_depth(depth_list.decode())
    dep_img = dep_img / 1000.0 # 深度图以毫米为单位
    dep_img = aug_depth(dep_img)
    dep_img = tl.prepro.drop(dep_img, keep=np.random.uniform(0.5, 1.0))

    cam = cPickle.loads(cam)
    annos2d = list(cPickle.loads(ground_truth2d))[:n_pos]
    annos2d = np.array(annos2d)
    annos3d = list(cPickle.loads(ground_truth3d))[:n_pos]
    annos3d = np.array(annos3d) / 100.0 # 三维点坐标以厘米为单位

    # create voxel occupancy grid from the warped depth map
    voxel_grid, voxel_coords2d, voxel_coordsvis, trafo_params = create_voxelgrid(cam, dep_img, annos2d, (xdim, ydim, zdim), 1.2)
    voxel_coords3d = (annos3d - trafo_params['root']) / trafo_params['scale']

    # Augmentation of voxels and keypoints
    coords2d, coords3d, coordsvis = voxel_coords2d.tolist(), voxel_coords3d.tolist(), voxel_coordsvis.tolist()
    rotate_matrix = tl.prepro.transform_matrix_offset_center(tl.prepro.affine_rotation_matrix(angle=(-15, 15)), x=xdim, y=xdim)
    voxel_grid = tl.prepro.affine_transform_cv2(voxel_grid.transpose([1, 0, 2]), rotate_matrix).transpose([1, 0, 2])
    coords2d = keypoints_affine(coords2d, rotate_matrix)
    coords3d = keypoints_affine(coords3d, rotate_matrix)
    if np.random.uniform() > 0.5:
        voxel_grid = np.flip(voxel_grid, axis=0)
        coords2d, coordsvis = keypoint_flip(coords2d, (xdim, ydim), 0, coordsvis)
        coords3d, coordsvis = keypoint_flip(coords3d, (xdim, ydim, zdim), 0, coordsvis)
    voxel_coords2d, voxel_coords3d, voxel_coordsvis = np.array(coords2d), np.array(coords3d), np.array(coordsvis) 

    heatmap_kp, voxel_coordsvis = get_kp_heatmap(voxel_coords2d, (xdim, ydim), sigma, voxel_coordsvis)
    voxel_kp = np.tile(np.expand_dims(heatmap_kp, 2), [1, 1, zdim, 1])
    voxel_grid = np.expand_dims(voxel_grid, -1)
    input_3d = np.concatenate((voxel_grid, voxel_kp), 3)

    output_3d, voxel_coordsvis = get_3d_heatmap(voxel_coords3d, (xdim, ydim, zdim), sigma, voxel_coordsvis)
    mask_vis = np.ones_like(output_3d)*voxel_coordsvis

    input_3d = np.array(input_3d, dtype=np.float32)
    output_3d = np.array(output_3d, dtype=np.float32)
    mask_vis = np.array(mask_vis, dtype=np.float32)

    return input_3d, output_3d, mask_vis


def _map_fn(depth_list, cams, anno2ds, anno3ds):
    """TF Dataset pipeline."""
    input_3d, output_3d, mask_vis = tf.py_func(_3d_data_aug_fn, [depth_list,cams,anno2ds,anno3ds], [tf.float32, tf.float32, tf.float32])

    input_3d = tf.reshape(input_3d, [xdim, ydim, zdim, n_pos+1])
    output_3d = tf.reshape(output_3d, [xdim, ydim, zdim, n_pos])
    mask_vis = tf.reshape(mask_vis, [xdim, ydim, zdim, n_pos])

    return input_3d, output_3d, mask_vis


def train(training_dataset, epoch, n_step):
    ds = training_dataset.shuffle(buffer_size=4096)  # shuffle before loading images
    ds = ds.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count() // 2)  # decouple the heavy map_fn
    ds = ds.batch(batch_size)
    ds = ds.prefetch(2)
    iterator = ds.make_one_shot_iterator()
    one_element = iterator.get_next()
    head_net, head_loss = make_model(*one_element, reuse=False, use_slim=b_slim)
    stage_losses = head_net.stage_losses
    l2_loss = head_net.l2_loss

    new_lr_decay = lr_decay_factor**((epoch-1)*n_step // lr_decay_interval)
    print('Start - epoch: {} n_step: {} batch_size: {} lr_init: {} lr_decay_interval: {}'.format(
        epoch, n_step, batch_size, lr_init*new_lr_decay, lr_decay_interval))

    lr_v = tf.Variable(lr_init * new_lr_decay, trainable=False, name='learning_rate')
    global_step = tf.Variable(1, trainable=False)
    train_op = tf.train.AdamOptimizer(lr_v).minimize(head_loss, global_step=global_step)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    # start training
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # restore pre-trained weights
        try:
            tl.files.load_and_assign_npz_dict(sess=sess, name=os.path.join(model_path, 'voxelposenet.npz'))
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

            [_, _loss, _stage_losses, _l2] = sess.run([train_op, head_loss, stage_losses, l2_loss])

            lr = sess.run(lr_v)
            print('Training loss at iteration {} / {} is: {} Learning rate {:10e} l2_loss {:10e} Took: {}s'.format(
                step, n_step, _loss, lr, _l2, time.time() - tic))
            for ix, sl in enumerate(_stage_losses):
                print('Network#', ix, 'Loss:', sl)

            # save intermediate results and model
            if (step != 0) and (step % save_interval == 0):
                tl.files.save_npz_dict(head_net.all_params, os.path.join(model_path, 'voxelposenet-'+str(epoch)+'-'+str(step)+'.npz'), sess=sess)
                tl.files.save_npz_dict(head_net.all_params, os.path.join(model_path, 'voxelposenet.npz'), sess=sess)
            # training finished
            if step == n_step:
                tl.files.save_npz_dict(head_net.all_params, os.path.join(model_path, 'voxelposenet-'+str(epoch)+'.npz'), sess=sess)
                tl.files.save_npz_dict(head_net.all_params, os.path.join(model_path, 'voxelposenet.npz'), sess=sess)
                break


def evaluate(evaluating_dataset, epoch, n_step):
    ds = evaluating_dataset.shuffle(buffer_size=4096)  # shuffle before loading images
    ds = ds.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count() // 2)  # decouple the heavy map_fn
    ds = ds.batch(batch_size)
    ds = ds.prefetch(2)
    iterator = ds.make_one_shot_iterator()
    one_element = iterator.get_next()
    head_net, head_loss = make_model(*one_element, reuse=True, use_slim=b_slim)
    l2_loss = head_net.l2_loss

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    # start evaluating
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # restore pre-trained weights
        try:
            tl.files.load_and_assign_npz_dict(sess=sess, name=os.path.join(model_path, 'voxelposenet.npz'))
        except:
            print("no pre-trained model")

        # evaluate all test files
        step = 0
        sum_loss = 0.0
        invalid_count = 0
        while True:
            tic = time.time()
            [_loss, _l2] = sess.run([head_loss, l2_loss])

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
        ##               train.mat
        ##               val.mat
        ##               test.mat
        ##               anno_01_00000118.mat
        ##               color_01_00000118.jpg
        ##               depth_01_00000118.png
        ##           /KINECTNODE2
        ##           ...
        ## have a folder with many folders: (which is common in industry)
        root_list = tl.files.load_folder_list(path=config.DATA.data_path)
        train_depths_list, train_cams_list, train_anno2ds_list, train_anno3ds_list = [], [], [], []
        val_depths_list, val_cams_list, val_anno2ds_list, val_anno3ds_list = [], [], [], []
        for root in root_list:
            folder_list = tl.files.load_folder_list(path=root)
            for folder in folder_list:
                if config.DATA.image_path in folder:
                    _depths_list, _cams_list, _anno2ds_list, _anno3ds_list = get_pose_data_list(folder, config.DATA.train_anno, 9, 0.25)
                    train_depths_list.extend(_depths_list)
                    train_cams_list.extend(_cams_list)
                    train_anno2ds_list.extend(_anno2ds_list)
                    train_anno3ds_list.extend(_anno3ds_list)
                    _depths_list, _cams_list, _anno2ds_list, _anno3ds_list = get_pose_data_list(folder, config.DATA.val_anno, 9, 0.25)
                    val_depths_list.extend(_depths_list)
                    val_cams_list.extend(_cams_list)
                    val_anno2ds_list.extend(_anno2ds_list)
                    val_anno3ds_list.extend(_anno3ds_list)
        print("Total {} samples for training, {} samples for validation.".format(len(train_depths_list), len(val_depths_list)))
    
    # define data augmentation
    def train_ds_generator():
        """TF Dataset generator."""
        for _input_depth, _calib_cam, _target_anno2d, _target_anno3d in zip(train_depths_list, train_cams_list, train_anno2ds_list, train_anno3ds_list):
            yield _input_depth.encode('utf-8'), cPickle.dumps(_calib_cam), cPickle.dumps(_target_anno2d), cPickle.dumps(_target_anno3d)
    train_ds = tf.data.Dataset().from_generator(train_ds_generator, output_types=(tf.string, tf.string, tf.string, tf.string))

    def val_ds_generator():
        """TF Dataset generator."""
        for _input_depth, _calib_cam, _target_anno2d, _target_anno3d in zip(val_depths_list, val_cams_list, val_anno2ds_list, val_anno3ds_list):
            yield _input_depth.encode('utf-8'), cPickle.dumps(_calib_cam), cPickle.dumps(_target_anno2d), cPickle.dumps(_target_anno3d)
    eval_ds = tf.data.Dataset().from_generator(val_ds_generator, output_types=(tf.string, tf.string, tf.string, tf.string))

    for epoch in range(1, n_epoch+1):
        tf.reset_default_graph()
        train(train_ds, epoch, len(train_depths_list)//batch_size)
        tf.reset_default_graph()
        evaluate(eval_ds, epoch, len(val_depths_list)//batch_size)