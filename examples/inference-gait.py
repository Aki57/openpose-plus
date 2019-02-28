#!/usr/bin/env python3
import os
import sys
import time

import cv2
import numpy as np
import scipy.io as sio
import scipy.misc as sm
import tensorflow as tf
import tensorlayer as tl

from easydict import EasyDict as edict

sys.path.append('.')

from openpose_plus.inference.common import measure, plot_humans, plot_3d_person, read_2dfiles, read_3dfiles, tranform_keypoints2d
from openpose_plus.inference.estimator import TfPoseEstimator, Pose3DEstimator
from openpose_plus.models import get_base_model, get_head_model
from openpose_plus.utils import Camera, create_voxelgrid, get_kp_heatmap


def get_files(data_path, color_folder, depth_folder, camera_file):
    rgbs_list, deps_list = [], []
    cam_info = None
    if not tl.files.folder_exists(data_path):
        print("[skip] data_path is not found: {}".format(data_path))
        pass
    rgb_path = os.path.join(data_path, color_folder)
    dep_path = os.path.join(data_path, depth_folder)
    cam_file = os.path.join(data_path, camera_file)
    if tl.files.folder_exists(rgb_path):
        rgbs_list = tl.files.load_file_list(rgb_path, regx='\.jpg')
    else:
        print("[skip] data_path is not found: {}".format(data_path))
    if tl.files.folder_exists(dep_path):
        deps_list = tl.files.load_file_list(dep_path, regx='\.png')
    else:
        print("[skip] data_path is not found: {}".format(data_path))
    if os.path.exists(cam_file):
        params = [line.split(" : ")[1] for line in open(cam_file,"r")]
        K = np.array([[],[],[]])
        distCoef = np.array([])
        cam_info = Camera(K, distCoef)
    else:
        print("[skip] cam_file is not found: {}".format(data_path))
    print("Total number of own images found:", len(rgbs_list))
    return rgbs_list, deps_list, cam_info


def inference(base_model_name, base_npz_path, head_model_name, head_npz_path, rgb_files, dep_files, cam_info, plot):
    height, width, channel = (368, 432, 4)
    base_model_func = get_base_model(base_model_name)
    e_2d = measure(lambda: TfPoseEstimator(base_npz_path, base_model_func, (height, width, channel)), 'create TfPoseEstimator')

    x_size, y_size, z_size = (64, 64, 64)
    head_model_func = get_head_model(head_model_name)
    e_3d = measure(lambda: Pose3DEstimator(head_npz_path, head_model_func, (x_size, y_size, z_size), False), 'create Pose3DEstimator')

    time0 = time.time()
    for idx, (rgb_name, dep_name, cam_info) in enumerate(zip(rgb_files, dep_files, cam_info)):
        rgb_img = cv2.imread(rgb_name, cv2.IMREAD_COLOR)
        dep_img = sm.imread(dep_name).astype('float32')  # depth map warped into the color frame
        dep_img = np.expand_dims(dep_img, axis=2)
        input_2d = np.concatenate((rgb_img, dep_img), axis=2)
        init_h, init_w, _ = input_2d.shape
        input_2d = cv2.resize(input_2d, (width, height))
        input_2d, init_h, init_w = measure(lambda: read_2dfiles(rgb_name, dep_name, height, width), 'read_2dfiles')
        humans, heatMap, pafMap = measure(lambda: e_2d.inference(input_2d), 'e_2d.inference')
        print('got %d humans from %s' % (len(humans), rgb_name[:-4]))
        if plot:
            plot_humans(input_2d, heatMap, pafMap, humans, '%02d' % (idx + 1))

        coords_xyz, coords_xyz_conf = list(), list()
        for h in humans:
            coords2d, coords2d_conf, coords2d_vis = tranform_keypoints2d(h.body_parts, init_w, init_h)
            input_3d, trafo_params = measure(lambda: read_3dfiles(dep_name, cam_info, coords2d, coords2d_vis, x_size, y_size, z_size), 'read_3dfiles')
            coords3d, coords3d_conf = measure(lambda: e_3d.inference(input_3d), 'e_3d.inference')
            coords3d_pred = coords3d * trafo_params['scale'] + trafo_params['root']
            coords3d_pred_proj = cam_info.unproject(coords2d, coords3d_pred[:, -1])

            cond = coords2d_conf > coords3d_conf  # use backproj only when 2d was visible and 2d/3d roughly matches
            coords3d_pred[cond, :] = coords3d_pred_proj[cond, :]
            coords3d_conf[cond] = coords2d_conf[cond]
            if plot:
                plot_3d_person(rgb_name, dep_name, coords3d_pred, coords2d_vis, Camera(cam_info['K'], cam_info['distCoef']), idx)

            coords_xyz.append(coords3d_pred)
            coords_xyz_conf.append(coords3d_conf)

    mean = (time.time() - time0) / len(rgb_files)
    print('inference all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


def main():
    base_npz_path = 'models/openposenet.npz' # str, default='', help='path to npz', required=True
    base_model = 'hao28_experimental' # str, default='hao28_experimental', help='mobilenet | mobilenet2 | hao28_experimental'
    head_npz_path = 'models/voxelposenet-False.npz' # str, default='', help='path to npz', required=True
    head_model = 'voxelposenet' # str, default='voxelposenet', help='voxelposenet | pixelposenet'
    plot = True # bool, default=False, help='draw the results'

    rgb_files, dep_files, cam_info = get_files('data/cmu_dataset', 'color', 'depth', 'kinect.txt')
    inference(base_model, base_npz_path, head_model, head_npz_path, rgb_files, dep_files, cam_info, plot)


if __name__ == '__main__':
    measure(main)