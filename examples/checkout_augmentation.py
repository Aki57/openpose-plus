#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
import tensorlayer as tl

from easydict import EasyDict as edict

sys.path.append('.')

from openpose_plus.inference.common import measure
from openpose_plus.utils import Camera, PoseInfo, read_depth, create_voxelgrid, keypoints_affine, keypoint_flip


def get_files(data_path):
    sum_cams_list, sum_depths_list, sum_joint2d_list, sum_joint3d_list = [], [], [], []
    root_list = tl.files.load_folder_list(path=data_path)
    for root in root_list:
        folder_list = tl.files.load_folder_list(path=root)
        for folder in folder_list:
            if 'KINECTNODE' in folder:
                print("[x] Get pose data from {}".format(folder))
                _cams_list, _depths_list, _joint2d_list, _joint3d_list = PoseInfo(folder, 'meta.mat', 9, 0.25).get_3d_data_list()
                sum_cams_list.extend(_cams_list)
                sum_depths_list.extend(_depths_list)
                sum_joint2d_list.extend(_joint2d_list)
                sum_joint3d_list.extend(_joint3d_list)
    print("Total number of own images found:", len(sum_cams_list))
    return sum_cams_list, sum_depths_list, sum_joint2d_list, sum_joint3d_list


def check_augmentation(dep_files, cam_list, joint2d_list, joint3d_list):
    time0 = time.time()
    for _, (dep_name, cam_info, joints_2d, joints_3d) in enumerate(zip(dep_files, cam_list, joint2d_list, joint3d_list)):
        # Augmentation of depth image
        dep_img = read_depth(dep_name)
        dep_img = dep_img / 1000.0 # 深度图以毫米为单位
        dep_img = tl.prepro.drop(dep_img, keep=np.random.uniform(0.5, 1.0))
        # dep_img = argue_depth(dep_img)

        joints_2d = np.array(joints_2d[:18])
        joints_3d = np.array(joints_3d[:18]) / 100.0 # 三维点坐标以厘米为单位

        # create voxel occupancy grid from the warped depth map
        xdim, ydim, zdim = (64, 64, 64)
        voxel_grid, voxel_coords2d, voxel_coordsvis, trafo_params = create_voxelgrid(cam_info, dep_img, joints_2d, (xdim, ydim, zdim), 1.2)
        voxel_coords3d = (joints_3d - trafo_params['root']) / trafo_params['scale']
        plt

        # Augmentation of voxels and keypoints
        coords2d, coords3d, coordsvis = voxel_coords2d.tolist(), voxel_coords3d.tolist(), voxel_coordsvis.tolist()
        rotate_matrix = tl.prepro.transform_matrix_offset_center(tl.prepro.affine_rotation_matrix(angle=(-15, 15)), x=xdim, y=xdim)
        voxel_grid = tl.prepro.affine_transform(voxel_grid, rotate_matrix)
        coords2d = keypoints_affine(coords2d, rotate_matrix)
        coords3d = keypoints_affine(coords3d, rotate_matrix)
        if np.random.uniform() > 0.5:
            voxel_grid = np.flip(voxel_grid, axis=0)
            coords2d, coordsvis = keypoint_flip(coords2d, (xdim, ydim), 0, coordsvis)
            coords3d, coordsvis = keypoint_flip(coords3d, (xdim, ydim, zdim), 0, coordsvis)
        voxel_coords2d, voxel_coords3d, voxel_coordsvis = np.array(coords2d), np.array(coords3d), np.array(coordsvis) 

    mean = (time.time() - time0) / len(dep_files)
    print('Check all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


def main():
    _cam_list, _dep_files, _joint2d_list, _joint3d_list  = get_files('f:/Lab/dataset/panoptic-toolbox/data')
    check_augmentation(_dep_files, _cam_list, _joint2d_list, _joint3d_list)


if __name__ == '__main__':
    measure(main)