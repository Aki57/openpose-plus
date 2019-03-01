#!/usr/bin/env python3
import os
import sys
import time

import tensorflow as tf
import tensorlayer as tl

from easydict import EasyDict as edict

sys.path.append('.')

from openpose_plus.inference.common import measure, plot_humans, plot_human3d, read_2dfiles, read_3dfiles, tranform_keypoints2d
from openpose_plus.inference.estimator import TfPoseEstimator, Pose3DEstimator
from openpose_plus.models import get_base_model, get_head_model
from openpose_plus.utils import Camera, PoseInfo, create_voxelgrid, get_kp_heatmap


def get_files(data_path):
    sum_rgbs_list, sum_depths_list, sum_cams_list = [], [], []
    root_list = tl.files.load_folder_list(path=data_path)
    for root in root_list:
        folder_list = tl.files.load_folder_list(path=root)
        for folder in folder_list:
            if 'KINECTNODE' in folder:
                print("[x] Get pose data from {}".format(folder))
                _cams_list, _rgbs_list, _depths_list = PoseInfo(folder, 'meta.mat', 9, 0.25).get_test_data_list()
                sum_cams_list.extend(_cams_list)
                sum_rgbs_list.extend(_rgbs_list)
                sum_depths_list.extend(_depths_list)
    print("Total number of own images found:", len(sum_rgbs_list))
    return sum_rgbs_list, sum_depths_list, sum_cams_list


def checkfiles(rgb_files, dep_files, cam_list):
    time0 = time.time()
    for idx, (rgb_name, dep_name, cam_info) in enumerate(zip(rgb_files, dep_files, cam_list)):
        import scipy.io as sio
        import numpy as np
        import cv2
        try:
            rgb_img = cv2.imread(rgb_name, cv2.IMREAD_COLOR)
        except:
            print("The rgb file is broken : {}".format(rgb_name))
        try:
            dep_img = sio.loadmat(dep_name)['depthim_incolor']
            dep_img = dep_img / 1000.0 # 深度图以毫米为单位
            dep_img = tl.prepro.drop(dep_img, keep=np.random.uniform(0.5, 1.0))
        except:
            print("The depth file is broken : {}".format(dep_name))
        try:
            cam = Camera(cam_info['K'], cam_info['distCoef'])
        except:
            print("The cam info is broken : {}".format(dep_name))

    mean = (time.time() - time0) / len(rgb_files)
    print('inference all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


def main():
    _rgb_files, _dep_files, _cam_list = get_files('data/cmu_dataset')
    checkfiles(_rgb_files, _dep_files, _cam_list)


if __name__ == '__main__':
    measure(main)