#!/usr/bin/env python3
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from easydict import EasyDict as edict

sys.path.append('.')

from openpose_plus.inference.common import measure, do_plot, plot_humans, plot_human2d, plot_human3d, read_2dfiles, read_3dfiles, tranform_keypoints2d
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


def write_gait(coords_list, filename):
    with open(filename,'w') as f:
        # 对openpose输出结果的转换
        transform = [8, 8, 1, 0, 5, 6, 7, -1, 2, 3, 4, -1, 11, 12, 13, -1, 8, 9, 10, -1, 2, -1, -1, -1, -1]
                    #[11 11______________________________________________________________5________________]
                    #[___1________________________________________________________________________________]
        for coords in coords_list:
            new_coords = []
            if coords is None:
                new_coords.append(np.array([50000,50000,50000]))
                new_coords = new_coords*25
            elif coords.shape[1] == 2:
                coords_z = 50000*np.ones_like(coords[:,0])
                coords = np.concatenate((coords, np.expand_dims(coords_z,-1)), axis=1)
            for idx in transform:
                new_coords.append(coords[idx,:] if idx >= 0 else np.array([50000,50000,50000]))
            new_coords[0] = (coords[8,:] + coords[11,:])/2
            new_coords[1] = (new_coords[0] + coords[1,:])/2
            new_coords[20] = (coords[2,:] + coords[5,:])/2

            line = ''
            for coord in new_coords:
                coord = np.array(coord).astype('int')
                line += str(coord[0]) + ' ' + str(coord[1]) + ' ' + str(coord[2]) + ' '
            line += '\n'
            f.writelines(line)


def inference_data(base_model_name, base_npz_path, head_model_name, head_npz_path, rgb_files, dep_files, cam_info):
    height, width, channel = (368, 432, 4)
    base_model_func = get_base_model(base_model_name)
    e_2d = measure(lambda: TfPoseEstimator(base_npz_path, base_model_func, (height, width, channel)), 'create TfPoseEstimator')

    x_size, y_size, z_size = (64, 64, 64)
    head_model_func = get_head_model(head_model_name)
    e_3d = measure(lambda: Pose3DEstimator(head_npz_path, head_model_func, (x_size, y_size, z_size), False), 'create Pose3DEstimator')

    time0 = time.time()
    coords_uv_list, coords_xyz_list = list(), list()
    for _, (rgb_name, dep_name) in enumerate(zip(rgb_files, dep_files)):
        input_2d, init_h, init_w = measure(lambda: read_2dfiles(rgb_name, dep_name, height, width), 'read_2dfiles')
        humans, _, _ = measure(lambda: e_2d.inference(input_2d), 'e_2d.inference')
        print('got %d humans from %s' % (len(humans), rgb_name[:-4]))

        if len(humans) is 0:
            coords_uv_list.append(None)
            coords_xyz_list.append(None)
        else:
            coords2d, coords2d_conf, coords2d_vis = tranform_keypoints2d(humans[0].body_parts, init_w, init_h, 0.1)
            input_3d, trafo_params = measure(lambda: read_3dfiles(dep_name, cam_info, coords2d, coords2d_vis, x_size, y_size, z_size), 'read_3dfiles')
            coords3d, coords3d_conf = measure(lambda: e_3d.inference(input_3d), 'e_3d.inference')
            coords3d_pred = coords3d * trafo_params['scale'] + trafo_params['root']
            coords3d_pred_proj = Camera(cam_info['K'], cam_info['distCoef']).unproject(coords2d, coords3d_pred[:, -1])

            cond = coords2d_conf > coords3d_conf  # use backproj only when 2d was visible and 2d/3d roughly matches
            coords3d_pred[cond, :] = coords3d_pred_proj[cond, :]
            coords3d_conf[cond] = coords2d_conf[cond]

            coords_uv_list.append(coords2d)
            coords_xyz_list.append(coords3d_pred*1000.0)
            
    write_gait(coords_uv_list, 'gait2d.txt')
    write_gait(coords_xyz_list, 'gait3d.txt')

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
        print('got %d humans from %s' % (len(humans), rgb_name[:-4]))
        plot_humans(input_2d, heatMap, pafMap, humans, '%02d' % (idx + 1))

        if len(humans):
            coords2d, _, coords2d_vis = tranform_keypoints2d(humans[0].body_parts, init_w, init_h, 0.1)
            plot_human2d(rgb_name, dep_name, coords2d, idx, coords2d_vis)

        do_plot()

    mean = (time.time() - time0) / len(rgb_files)
    print('inference all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


def inference_3d(base_model_name, base_npz_path, head_model_name, head_npz_path, rgb_files, dep_files, cam_info):
    height, width, channel = (368, 432, 4)
    base_model_func = get_base_model(base_model_name)
    e_2d = measure(lambda: TfPoseEstimator(base_npz_path, base_model_func, (height, width, channel)), 'create TfPoseEstimator')

    x_size, y_size, z_size = (64, 64, 64)
    head_model_func = get_head_model(head_model_name)
    e_3d = measure(lambda: Pose3DEstimator(head_npz_path, head_model_func, (x_size, y_size, z_size), False), 'create Pose3DEstimator')

    cam_calib =  Camera(cam_info['K'], cam_info['distCoef'])

    time0 = time.time()
    for idx, (rgb_name, dep_name) in enumerate(zip(rgb_files, dep_files)):
        input_2d, init_h, init_w = measure(lambda: read_2dfiles(rgb_name, dep_name, height, width), 'read_2dfiles')
        humans, _, _ = measure(lambda: e_2d.inference(input_2d), 'e_2d.inference')
        print('got %d humans from %s' % (len(humans), rgb_name[:-4]))

        if len(humans):
            coords2d, coords2d_conf, coords2d_vis = tranform_keypoints2d(humans[0].body_parts, init_w, init_h, 0.1)
            input_3d, trafo_params = measure(lambda: read_3dfiles(dep_name, cam_info, coords2d, coords2d_vis, x_size, y_size, z_size), 'read_3dfiles')

            coords3d, coords3d_conf = measure(lambda: e_3d.inference(input_3d), 'e_3d.inference')
            coords3d_pred = coords3d * trafo_params['scale'] + trafo_params['root']
            coords3d_pred_proj = cam_calib.unproject(coords2d, coords3d_pred[:, -1])
            cond = coords2d_conf > coords3d_conf  # use backproj only when 2d was visible and 2d/3d roughly matches
            coords3d_pred[cond, :] = coords3d_pred_proj[cond, :]
            coords3d_conf[cond] = coords2d_conf[cond]

            # coords3d = measure(lambda: e_3d.regression(input_3d), 'e_3d.inference')
            # coords3d_pred = coords3d * trafo_params['scale'] + trafo_params['root']

            plot_human3d(rgb_name, dep_name, coords3d_pred, cam_calib, idx, coords2d_vis)

        do_plot()

    mean = (time.time() - time0) / len(rgb_files)
    print('inference all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


if __name__ == '__main__':
    base_npz_path = 'models/2d/hao28-pose-average.npz' # str, default='', help='path to npz', required=True
    base_model = 'hao28_experimental' # str, default='hao28_experimental', help='mobilenet | mobilenet2 | hao28_experimental'
    head_npz_path = 'models/3d/voxelposenet-False.npz' # str, default='', help='path to npz', required=True
    head_model = 'voxelposenet' # str, default='voxelposenet', help='voxelposenet | pixelposenet'

    rgb_files, dep_files, cam_info = get_files('f:/Lab/dataset/Gait-Database', '2color', '2depth/filter', 'kinect2.txt')
    # inference_2d(base_model, base_npz_path, rgb_files, dep_files)
    inference_3d(base_model, base_npz_path, head_model, head_npz_path, rgb_files, dep_files, cam_info)
    # inference_data(base_model, base_npz_path, head_model, head_npz_path, rgb_files, dep_files, cam_info)