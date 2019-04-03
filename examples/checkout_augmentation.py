#!/usr/bin/env python3
import os
import sys
import time

import cv2
import numpy as np
import tensorlayer as tl

import matplotlib.pyplot as plt
from easydict import EasyDict as edict

sys.path.append('.')

from openpose_plus.inference.common import measure, CocoPairs
from openpose_plus.utils import Camera, PoseInfo, read_depth, aug_depth, create_voxelgrid, keypoints_affine, keypoint_flip

def get_2d_files(data_path):
    sum_rgbs_list, sum_depths_list, sum_joint2d_list = [], [], []
    root_list = tl.files.load_folder_list(path=data_path)
    for root in root_list:
        folder_list = tl.files.load_folder_list(path=root)
        for folder in folder_list:
            if 'KINECTNODE' in folder:
                print("[x] Get pose data from {}".format(folder))
                _rgbs_list, _depths_list, _joint2d_list = PoseInfo(folder, 'meta2.mat', 5, 0.25).get_2d_data_list()
                sum_rgbs_list.extend(_rgbs_list)
                sum_depths_list.extend(_depths_list)
                sum_joint2d_list.extend(_joint2d_list)
    print("Total number of own images found:", len(sum_rgbs_list))
    return sum_rgbs_list, sum_depths_list, sum_joint2d_list


def get_3d_files(data_path):
    sum_cams_list, sum_depths_list, sum_joint2d_list, sum_joint3d_list = [], [], [], []
    root_list = tl.files.load_folder_list(path=data_path)
    for root in root_list:
        folder_list = tl.files.load_folder_list(path=root)
        for folder in folder_list:
            if 'KINECTNODE' in folder:
                print("[x] Get pose data from {}".format(folder))
                _cams_list, _depths_list, _joint2d_list, _joint3d_list = PoseInfo(folder, 'meta2.mat', 9, 0.25).get_3d_data_list()
                sum_cams_list.extend(_cams_list)
                sum_depths_list.extend(_depths_list)
                sum_joint2d_list.extend(_joint2d_list)
                sum_joint3d_list.extend(_joint3d_list)
    print("Total number of own images found:", len(sum_cams_list))
    return sum_cams_list, sum_depths_list, sum_joint2d_list, sum_joint3d_list


def _show_input_2d(rgb_image, depth_image, coord_uv_list):
    import matplotlib.pyplot as plt
    plt.figure(figsize=[8, 7])
    plt.subplot(2,1,1)
    plt.imshow(rgb_image)
    for coord_uv in coord_uv_list:
        coord_uv = np.array(coord_uv)
        plt.plot(coord_uv[:,0], coord_uv[:,1], '.')
        for pair in CocoPairs:
            plt.plot(coord_uv[pair,0], coord_uv[pair,1], linewidth=2)
    plt.draw()
    
    plt.subplot(2,1,2)
    plt.imshow(depth_image)
    for coord_uv in coord_uv_list:
        coord_uv = np.array(coord_uv)
        plt.plot(coord_uv[:,0], coord_uv[:,1], 'r.')
        for pair in CocoPairs:
            plt.plot(coord_uv[pair,0], coord_uv[pair,1], linewidth=2)
    plt.draw()
    plt.show()


def _show_squeeze_pcl(voxel_pcl, coord_uv_vox):
    import matplotlib.pyplot as plt
    xdim, ydim, _ = voxel_pcl.shape
    min_voxel_pcl = np.zeros((xdim, ydim))
    for i in range(0,xdim):
        for j in range(0,ydim):
            min_pos = np.where(voxel_pcl[i,j,:] > 0)
            if min_pos[0].shape[0] > 0:
                min_voxel_pcl[i,j] = min_pos[0][0]

    plt.imshow(min_voxel_pcl.transpose([1, 0]))
    plt.plot(coord_uv_vox[:,0], coord_uv_vox[:,1], 'r.')
    plt.draw()
    plt.show()


def check_2d_data_aug(rgb_files, dep_files, joint2d_list):
    time0 = time.time()
    for _, (rgb_name, dep_name, joints_2d) in enumerate(zip(rgb_files, dep_files, joint2d_list)):
        # Augmentation of depth image
        rgb_image = cv2.imread(rgb_name, cv2.IMREAD_COLOR)
        rgb_image = 255 - rgb_image if np.random.uniform() < 0.25 else rgb_image

        depth_image = read_depth(dep_name)
        depth_image = depth_image / np.random.uniform(20, 40)
        depth_image = tl.prepro.drop(depth_image, keep=np.random.uniform())
        _show_input_2d(rgb_image, depth_image, joints_2d)

        ## 2d data augmentation
        # random transfrom
        M_rotate = tl.prepro.affine_rotation_matrix(angle=(-40, 40))  # original paper: -40~40 -> -30~30
        M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.5, 1.1))  # original paper: 0.5~1.1 -> 0.5~0.8
        M_combined = M_rotate.dot(M_zoom)
        transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=rgb_image.shape[1], y=rgb_image.shape[0])
        rgb_image = tl.prepro.affine_transform_cv2(rgb_image, transform_matrix)
        depth_image = tl.prepro.affine_transform_cv2(depth_image, transform_matrix, border_mode='replicate')
        joints_2d = tl.prepro.affine_transform_keypoints(joints_2d, transform_matrix)
        _show_input_2d(rgb_image, depth_image, joints_2d)

        # random crop and flip
        hin, win = (368, 368)
        rgb_image, joints_2d, depth_image = tl.prepro.keypoint_random_flip(rgb_image, joints_2d, depth_image, prob=0.5)    
        rgb_image, joints_2d, depth_image = tl.prepro.keypoint_resize_random_crop(rgb_image, joints_2d, depth_image, size=(hin, win)) # hao add
        _show_input_2d(rgb_image, depth_image, joints_2d)

    mean = (time.time() - time0) / len(dep_files)
    print('Check all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


def check_3d_data_aug(dep_files, cam_list, joint2d_list, joint3d_list):
    time0 = time.time()
    for _, (dep_name, cam_info, joints_2d, joints_3d) in enumerate(zip(dep_files, cam_list, joint2d_list, joint3d_list)):
        # Augmentation of depth image
        dep_img = read_depth(dep_name)
        dep_img = dep_img / 1000.0 # 深度图以毫米为单位
        dep_img = aug_depth(dep_img)
        dep_img = tl.prepro.drop(dep_img, keep=np.random.uniform(0.5, 1.0))

        joints_2d = np.array(joints_2d[:18])
        joints_3d = np.array(joints_3d[:18]) / 100.0 # 三维点坐标以厘米为单位

        # create voxel occupancy grid from the warped depth map
        xdim, ydim, zdim = (64, 64, 64)
        voxel_grid, voxel_coords2d, voxel_coordsvis, trafo_params = create_voxelgrid(cam_info, dep_img, joints_2d, (xdim, ydim, zdim), 1.2)
        voxel_coords3d = (joints_3d - trafo_params['root']) / trafo_params['scale']
        _show_squeeze_pcl(voxel_grid, voxel_coords2d)

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
        _show_squeeze_pcl(voxel_grid, voxel_coords2d)

    mean = (time.time() - time0) / len(dep_files)
    print('Check all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))


def main():
    # _rgb_files, _dep_files, _joint2d_list = get_2d_files('f:/Lab/dataset/panoptic-toolbox/data')
    # check_2d_data_aug(_rgb_files, _dep_files, _joint2d_list)

    _cam_list, _dep_files, _joint2d_list, _joint3d_list = get_3d_files('f:/Lab/dataset/panoptic-toolbox/data')
    check_3d_data_aug(_dep_files, _cam_list, _joint2d_list, _joint3d_list)


if __name__ == '__main__':
    measure(main)