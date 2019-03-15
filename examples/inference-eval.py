#!/usr/bin/env python3
import os
import sys
import time

import tensorflow as tf
import tensorlayer as tl

from easydict import EasyDict as edict

sys.path.append('.')

from openpose_plus.inference.common import measure, do_plot, plot_humans, plot_human2d, plot_human3d, read_2dfiles, read_3dfiles, tranform_keypoints2d
from openpose_plus.inference.estimator import TfPoseEstimator, Pose3DEstimator
from openpose_plus.models import get_base_model, get_head_model
from openpose_plus.utils import Camera, PoseInfo, create_voxelgrid, get_kp_heatmap


def get_files(data_path):
    sum_rgbs_list, sum_depths_list, sum_cams_list, sum_joint3d_list = [], [], [], []
    root_list = tl.files.load_folder_list(path=data_path)
    for root in root_list:
        folder_list = tl.files.load_folder_list(path=root)
        for folder in folder_list:
            if 'KINECTNODE' in folder:
                print("[x] Get pose data from {}".format(folder))
                _cams_list, _rgbs_list, _depths_list, _joint3d_list = PoseInfo(folder, 'meta.mat', 9, 0.25).get_val_data_list()
                sum_cams_list.extend(_cams_list)
                sum_rgbs_list.extend(_rgbs_list)
                sum_depths_list.extend(_depths_list)
                sum_joint3d_list.extend(_joint3d_list)
    print("Total number of own images found:", len(sum_rgbs_list))
    return sum_rgbs_list, sum_depths_list, sum_cams_list, sum_joint3d_list


def inference_data(base_model_name, base_npz_path, head_model_name, head_npz_path, rgb_files, dep_files, cam_list):
    height, width, channel = (368, 432, 4)
    base_model_func = get_base_model(base_model_name)
    e_2d = measure(lambda: TfPoseEstimator(base_npz_path, base_model_func, (height, width, channel)), 'create TfPoseEstimator')

    x_size, y_size, z_size = (64, 64, 64)
    head_model_func = get_head_model(head_model_name)
    e_3d = measure(lambda: Pose3DEstimator(head_npz_path, head_model_func, (x_size, y_size, z_size), False), 'create Pose3DEstimator')

    time0 = time.time()
    coords_uv_list, coords_xyz_list = list(), list()
    for _, (rgb_name, dep_name, cam_info) in enumerate(zip(rgb_files, dep_files, cam_list)):
        input_2d, init_h, init_w = measure(lambda: read_2dfiles(rgb_name, dep_name, height, width), 'read_2dfiles')
        humans, _, _ = measure(lambda: e_2d.inference(input_2d), 'e_2d.inference')
        print('got %d humans from %s' % (len(humans), rgb_name[:-4]))

        for pred_2d in humans:
            coords2d, coords2d_conf, coords2d_vis = tranform_keypoints2d(pred_2d.body_parts, init_w, init_h)
            input_3d, trafo_params = measure(lambda: read_3dfiles(dep_name, cam_info, coords2d, coords2d_vis, x_size, y_size, z_size), 'read_3dfiles')
            coords3d, coords3d_conf = measure(lambda: e_3d.inference(input_3d), 'e_3d.inference')
            coords3d_pred = coords3d * trafo_params['scale'] + trafo_params['root']
            coords3d_pred_proj = Camera(cam_info['K'], cam_info['distCoef']).unproject(coords2d, coords3d_pred[:, -1])

            cond = coords2d_conf > coords3d_conf  # use backproj only when 2d was visible and 2d/3d roughly matches
            coords3d_pred[cond, :] = coords3d_pred_proj[cond, :]
            coords3d_conf[cond] = coords2d_conf[cond]
            
            coords_uv_list.append(coords2d)
            coords_xyz_list.append(coords3d_pred*100.0)

    # TODO: 输出评估数据

    mean = (time.time() - time0) / len(rgb_files)
    print('inference all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


def inference_2d(base_model_name, base_npz_path, rgb_files, dep_files):
    height, width, channel = (368, 432, 4)
    base_model_func = get_base_model(base_model_name)
    e_2d = measure(lambda: TfPoseEstimator(base_npz_path, base_model_func, (height, width, channel)), 'create TfPoseEstimator')

    time0 = time.time()
    for idx, (rgb_name, dep_name) in enumerate(zip(rgb_files, dep_files)):
        input_2d, init_h, init_w = measure(lambda: read_2dfiles(rgb_name, dep_name, height, width), 'read_2dfiles')
        humans, heatMap, pafMap = measure(lambda: e_2d.inference(input_2d), 'e_2d.inference')
        plot_humans(input_2d, heatMap, pafMap, humans, '%02d' % (idx + 1))

        for h, pred_2d in enumerate(humans):
            coords2d, _, coords2d_vis = tranform_keypoints2d(pred_2d.body_parts, init_w, init_h, 0.1)
            plot_human2d(rgb_name, dep_name, coords2d, h, coords2d_vis)

        do_plot()

    mean = (time.time() - time0) / len(rgb_files)
    print('inference all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


def inference_3d(base_model_name, base_npz_path, head_model_name, head_npz_path, rgb_files, dep_files, cam_list, joint3d_list):
    height, width, channel = (368, 432, 4)
    base_model_func = get_base_model(base_model_name)
    e_2d = measure(lambda: TfPoseEstimator(base_npz_path, base_model_func, (height, width, channel)), 'create TfPoseEstimator')

    x_size, y_size, z_size = (64, 64, 64)
    head_model_func = get_head_model(head_model_name)
    e_3d = measure(lambda: Pose3DEstimator(head_npz_path, head_model_func, (x_size, y_size, z_size), False), 'create Pose3DEstimator')

    time0 = time.time()
    for idx, (rgb_name, dep_name, cam_info, joints3d) in enumerate(zip(rgb_files, dep_files, cam_list, joint3d_list)):
        input_2d, init_h, init_w = measure(lambda: read_2dfiles(rgb_name, dep_name, height, width), 'read_2dfiles')
        humans, _, _ = measure(lambda: e_2d.inference(input_2d), 'e_2d.inference')
        print('got %d humans from %s' % (len(humans), rgb_name[:-4]))

        cam_calib =  Camera(cam_info['K'], cam_info['distCoef'])
        for h, (pred_2d, gt_3d) in enumerate(zip(humans, joints3d)):
            coords2d, coords2d_conf, coords2d_vis = tranform_keypoints2d(pred_2d.body_parts, init_w, init_h)
            input_3d, trafo_params = measure(lambda: read_3dfiles(dep_name, cam_info, coords2d, coords2d_vis, x_size, y_size, z_size), 'read_3dfiles')
            coords3d, coords3d_conf = measure(lambda: e_3d.inference(input_3d), 'e_3d.inference')
            coords3d_pred = coords3d * trafo_params['scale'] + trafo_params['root']
            coords3d_pred_proj = cam_calib.unproject(coords2d, coords3d_pred[:, -1])

            cond = coords2d_conf > coords3d_conf  # use backproj only when 2d was visible and 2d/3d roughly matches
            coords3d_pred[cond, :] = coords3d_pred_proj[cond, :]
            coords3d_conf[cond] = coords2d_conf[cond]
            plot_human3d(rgb_name, dep_name, coords3d_pred, cam_calib, 'pre%d'%h, coords2d_vis)
            plot_human3d(rgb_name, dep_name, gt_3d[:18], Camera(cam_info['K'], cam_info['distCoef']), 'gt%d'%h)

        do_plot()

    mean = (time.time() - time0) / len(rgb_files)
    print('inference all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


if __name__ == '__main__':
    base_npz_path = 'models/2d/openposenet-1.npz' # str, default='', help='path to npz', required=True
    base_model = 'hao28_experimental' # str, default='hao28_experimental', help='mobilenet | mobilenet2 | hao28_experimental'
    head_npz_path = 'models/3d/voxelposenet-False.npz' # str, default='', help='path to npz', required=True
    head_model = 'voxelposenet' # str, default='voxelposenet', help='voxelposenet | pixelposenet'
    interval = 100 # int, default=1, help='sample times of images.'

    _rgb_files, _dep_files, _cam_list, _joint3d_list  = get_files('f:/Lab/dataset/panoptic-toolbox/data')
    rgb_files = (_rgb_files)[::interval] # list of str, default='', help='comma separate list of image filenames', required=True
    dep_files = (_dep_files)[::interval] # list of str, default='', help='comma separate list of depth filenames', required=True
    cam_list = (_cam_list)[::interval] # list of str, default='', help='comma separate list of cam infos', required=True
    joint3d_list = (_joint3d_list)[::interval] # list of str, default='', help='comma separate list of cam infos', required=True

    # inference_2d(base_model, base_npz_path, rgb_files, dep_files)
    inference_3d(base_model, base_npz_path, head_model, head_npz_path, rgb_files, dep_files, cam_list, joint3d_list)
    # inference_data(base_model, base_npz_path, head_model, head_npz_path, rgb_files, dep_files, cam_list)