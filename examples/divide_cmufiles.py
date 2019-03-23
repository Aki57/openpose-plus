#!/usr/bin/env python3
import os
import sys
import time

import scipy.io as sio
import tensorlayer as tl

from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split

sys.path.append('.')

from openpose_plus.inference.common import measure


def split_meta(base_dir, meta_name, train_name, val_name, test_name):
    """
    Divide Meta file into train mata and test meta.
    Skip missing metas.
    """
    meta_path = os.path.join(base_dir, meta_name)
    train_path = os.path.join(base_dir, train_name)
    val_path = os.path.join(base_dir, val_name)
    test_path = os.path.join(base_dir, test_name)

    if os.path.exists(meta_path):
        cam_info = sio.loadmat(meta_path)['cam'][0]
        rgb_names = sio.loadmat(meta_path)['rgb'][0]
        depth_names = sio.loadmat(meta_path)['depth'][0]
        anno_names = sio.loadmat(meta_path)['anno'][0]
        print("{} has {} groups of images".format(base_dir, len(anno_names)))

        train_rgb_names, _rgb_names, \
        train_depth_names, _depth_names, \
        train_anno_names, _anno_names = train_test_split(rgb_names, depth_names, anno_names, test_size=0.2)
        val_rgb_names, test_rgb_names, \
        val_depth_names, test_depth_names, \
        val_anno_names, test_anno_names = train_test_split(_rgb_names, _depth_names, _anno_names, test_size=0.5)
        print("{} samples for training, {} for validation, {} for test.".format(len(train_anno_names), len(val_anno_names), len(test_anno_names)))

        train_rgb_names.sort()
        train_depth_names.sort()
        train_anno_names.sort()
        sio.savemat(train_path, {'cam':cam_info,'rgb':train_rgb_names,'depth':train_depth_names,'anno':train_anno_names})

        val_rgb_names.sort()
        val_depth_names.sort()
        val_anno_names.sort()
        sio.savemat(val_path, {'cam':cam_info,'rgb':val_rgb_names,'depth':val_depth_names,'anno':val_anno_names})

        test_rgb_names.sort()
        test_depth_names.sort()
        test_anno_names.sort()
        sio.savemat(test_path, {'cam':cam_info,'rgb':test_rgb_names,'depth':test_depth_names,'anno':test_anno_names})
        print("Dataset division is finished in ", base_dir)
    else:
        print("[skip] meta.mat is not found: ", base_dir)


def main():
    time0 = time.time()

    root_list = tl.files.load_folder_list(path='f:/Lab/dataset/panoptic-toolbox/data')
    for root in root_list:
        folder_list = tl.files.load_folder_list(path=root)
        for folder in folder_list:
            if 'KINECTNODE' in folder:
                print("[x] Get pose data from {}".format(folder))
                split_meta(folder, 'meta.mat', 'train.mat', 'val.mat', 'test.mat')

    print('Check all took: %f' % (time.time() - time0))


if __name__ == '__main__':
    measure(main)