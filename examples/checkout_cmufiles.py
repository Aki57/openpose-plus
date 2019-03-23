#!/usr/bin/env python3
import os
import sys
import time

import tensorlayer as tl

from easydict import EasyDict as edict

sys.path.append('.')

from openpose_plus.inference.common import measure
from openpose_plus.utils import Camera, PoseInfo


def get_files(data_path):
    sum_rgbs_list, sum_depths_list, sum_cams_list = [], [], []
    root_list = tl.files.load_folder_list(path=data_path)
    for root in root_list:
        folder_list = tl.files.load_folder_list(path=root)
        for folder in folder_list:
            if 'KINECTNODE' in folder:
                print("[x] Get pose data from {}".format(folder))
                _cams_list, _rgbs_list, _depths_list, _ = PoseInfo(folder, 'meta.mat', 9, 0.25).get_val_data_list()
                sum_cams_list.extend(_cams_list)
                sum_rgbs_list.extend(_rgbs_list)
                sum_depths_list.extend(_depths_list)
    print("Total number of own images found:", len(sum_rgbs_list))
    return sum_rgbs_list, sum_depths_list, sum_cams_list


def check_files(rgb_files, dep_files, cam_list):
    time0 = time.time()
    for _, (rgb_name, dep_name, cam_info) in enumerate(zip(rgb_files, dep_files, cam_list)):
        import scipy.io as sio
        import numpy as np
        import cv2
        try:
            rgb_img = cv2.imread(rgb_name, cv2.IMREAD_COLOR)
        except:
            print("The rgb file is broken : {}".format(rgb_name))
        try:
            dep_img = sio.loadmat(dep_name)['depthim_incolor']
        except:
            print("The depth file is broken : {}".format(dep_name))
        try:
            cam = Camera(cam_info['K'], cam_info['distCoef'])
        except:
            print("The cam info is broken : {}".format(dep_name))

    mean = (time.time() - time0) / len(rgb_files)
    print('Check all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


def main():
    _rgb_files, _dep_files, _cam_list = get_files('f:/Lab/dataset/panoptic-toolbox/data')
    check_files(_rgb_files, _dep_files, _cam_list)


if __name__ == '__main__':
    measure(main)