#!/usr/bin/env python3
import os
import sys
import time

import cv2
import numpy as np
import scipy.misc as sm
import tensorflow as tf
import tensorlayer as tl

from easydict import EasyDict as edict

sys.path.append('.')

from openpose_plus.inference.common import measure, plot_humans, plot_3d_gait, read_2dfiles, read_3dfiles, tranform_keypoints2d
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
        rgbs_list = [rgb_path + '/' + rgb for rgb in rgbs_list]
    else:
        print("[skip] rgb_path is not found: {}".format(data_path))
    if tl.files.folder_exists(dep_path):
        deps_list = tl.files.load_file_list(dep_path, regx='\.png')
        deps_list = [dep_path + '/' + dep for dep in deps_list]
    else:
        print("[skip] dep_path is not found: {}".format(data_path))
    if os.path.exists(cam_file):
        params = [float(line.split(" : ")[1]) for line in open(cam_file,"r")]
        cam_K = np.array([[params[0], 0.0, params[2]],
                        [0.0, params[1], params[3]],
                        [0.0, 0.0, 1.0]])
        cam_distCoef = np.array(params[4:])
        cam_info = dict(zip(['K', 'distCoef'], [cam_K, cam_distCoef]))
    else:
        print("[skip] cam_file is not found: {}".format(data_path))
    print("Total number of own images found:", len(rgbs_list))
    return rgbs_list, deps_list, cam_info


def write_gait(coords_xyz_list, filename):
    with open(filename,'w') as f:
        # 对openpose输出结果的转换
        transform = [8, 8, 1, 0, 5, 6, 7, -1, 2, 3, 4, -1, 11, 12, 13, -1, 8, 9, 10, -1, 2, -1, -1, -1, -1]
                    #[11 11______________________________________________________________5________________]
                    #[___1________________________________________________________________________________]
        for coords_xyz in coords_xyz_list:
            new_coords_xyz = []
            if coords_xyz is None:
                new_coords_xyz.append(np.array([50000,50000,50000]))
                new_coords_xyz = new_coords_xyz*25
            else:
                coords_xyz *= 1000.0
                for idx in transform:
                    new_coords_xyz.append(coords_xyz[idx,:] if idx >= 0 else np.array([50000,50000,50000]))
                new_coords_xyz[0] = (coords_xyz[8,:] + coords_xyz[11,:])/2
                new_coords_xyz[1] = (new_coords_xyz[0] + coords_xyz[1,:])/2
                new_coords_xyz[20] = (coords_xyz[2,:] + coords_xyz[5,:])/2

            line = ''
            for coord in new_coords_xyz:
                coord = np.array(coord).astype('int')
                line += str(coord[0]) + ' ' + str(coord[1]) + ' ' + str(coord[2]) + ' '
            line += '\n'
            f.writelines(line)


def inference(base_model_name, base_npz_path, head_model_name, head_npz_path, rgb_files, dep_files, cam_info, plot):
    height, width, channel = (368, 432, 4)
    base_model_func = get_base_model(base_model_name)
    e_2d = measure(lambda: TfPoseEstimator(base_npz_path, base_model_func, (height, width, channel)), 'create TfPoseEstimator')

    x_size, y_size, z_size = (64, 64, 64)
    head_model_func = get_head_model(head_model_name)
    e_3d = measure(lambda: Pose3DEstimator(head_npz_path, head_model_func, (x_size, y_size, z_size), False), 'create Pose3DEstimator')

    time0 = time.time()
    coords_xyz_list, coords_xyz_conf = list(), list()
    for idx, (rgb_name, dep_name) in enumerate(zip(rgb_files, dep_files)):
        init_w, init_h = (1920, 1080)
        rgb_img = cv2.imread(rgb_name, cv2.IMREAD_COLOR)
        rgb_img = cv2.resize(rgb_img, (init_w, init_h))
        dep_img = np.expand_dims(sm.imread(dep_name).astype('float32') / 20.0, axis=2)
        input_2d = np.concatenate((rgb_img, dep_img), axis=2)
        input_2d = cv2.resize(input_2d, (width, height)) / 255.0
        humans, heatMap, pafMap = measure(lambda: e_2d.inference(input_2d), 'e_2d.inference')
        print('got %d humans from %s' % (len(humans), rgb_name[:-4]))
        if plot:
            plot_humans(input_2d, heatMap, pafMap, humans, '%02d' % (idx + 1))

        if len(humans) is 0:
            coords_xyz_list.append(None)
            coords_xyz_conf.append(None)
        else:
            coords2d, coords2d_conf, coords2d_vis = tranform_keypoints2d(humans[0].body_parts, init_w, init_h)
            dep_img = sm.imread(dep_name).astype('float32') / 1000.0
            voxel_grid, voxel_coords2d, voxel_coordsvis, trafo_params = create_voxelgrid(cam_info, dep_img, coords2d, (x_size, y_size, z_size), 1.2, coords2d_vis)
            voxel_grid = np.expand_dims(voxel_grid, -1)
            voxel_kp, _ = get_kp_heatmap(voxel_coords2d, (x_size, y_size), 3.0, voxel_coordsvis)
            voxel_kp = np.tile(np.expand_dims(voxel_kp, 2), [1, 1, z_size, 1])
            input_3d = np.expand_dims(np.concatenate((voxel_grid, voxel_kp), 3), 0)
            coords3d, coords3d_conf = measure(lambda: e_3d.inference(input_3d), 'e_3d.inference')
            coords3d_pred = coords3d * trafo_params['scale'] + trafo_params['root']
            coords3d_pred_proj = Camera(cam_info['K'], cam_info['distCoef']).unproject(coords2d, coords3d_pred[:, -1])

            cond = coords2d_conf > coords3d_conf  # use backproj only when 2d was visible and 2d/3d roughly matches
            coords3d_pred[cond, :] = coords3d_pred_proj[cond, :]
            coords3d_conf[cond] = coords2d_conf[cond]
            coords_xyz_list.append(coords3d_pred)
            coords_xyz_conf.append(coords3d_conf)
            if plot:
                plot_3d_gait(rgb_name, dep_name, coords3d_pred, Camera(cam_info['K'], cam_info['distCoef']), idx, coords2d_vis)

    write_gait(coords_xyz_list, 'gait.txt')
    mean = (time.time() - time0) / len(rgb_files)
    print('inference all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


if __name__ == '__main__':
    base_npz_path = 'models/openposenet.npz' # str, default='', help='path to npz', required=True
    base_model = 'hao28_experimental' # str, default='hao28_experimental', help='mobilenet | mobilenet2 | hao28_experimental'
    head_npz_path = 'models/voxelposenet-1.npz' # str, default='', help='path to npz', required=True
    head_model = 'voxelposenet' # str, default='voxelposenet', help='voxelposenet | pixelposenet'
    plot = True # bool, default=False, help='draw the results'

    rgb_files, dep_files, cam_info = get_files('f:/Lab/dataset/Gait-Database', 'color', 'depth', 'kinect.txt')
    inference(base_model, base_npz_path, head_model, head_npz_path, rgb_files, dep_files, cam_info, plot)