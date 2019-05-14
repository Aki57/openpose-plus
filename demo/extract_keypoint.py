#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

sys.path.append('.')

from openpose_plus.inference.common import measure, read_2dfiles, read_3dfiles, tranform_keypoints2d
from openpose_plus.inference.estimator import TfPoseEstimator, Pose3DEstimator
from openpose_plus.models import get_base_model, get_head_model
from openpose_plus.utils import Camera, create_voxelgrid, get_kp_heatmap

class KeypointExtractor:

    def __init__(self, base_model_name, base_npz_path, head_model_name, head_npz_path, size_2d=(368, 656, 4), size_3d=(64, 64, 64)):
        # 可以将以下变量写成参数和类成员
        # base_npz_path = 'models/2d/hao28-pose-average.npz' # str, default='', help='path to npz', required=True
        # base_model_name = 'hao28_experimental' # str, default='hao28_experimental', help='mobilenet | mobilenet2 | hao28_experimental'
        # head_npz_path = 'models/3d/voxelposenet-False.npz' # str, default='', help='path to npz', required=True
        # head_model_name = 'voxelposenet' # str, default='voxelposenet', help='voxelposenet | pixelposenet'

        self.size_2d = size_2d
        self.size_3d = size_3d

        # 初始化模型, 可以继续拆分为类成员
        base_model_func = get_base_model(base_model_name)
        self.e_2d = measure(lambda: TfPoseEstimator(base_npz_path, base_model_func, self.size_2d), 'create TfPoseEstimator')

        head_model_func = get_head_model(head_model_name)
        self.e_3d = measure(lambda: Pose3DEstimator(head_npz_path, head_model_func, self.size_3d, False), 'create Pose3DEstimator')

    def get_files(self, data_path, color_folder, depth_folder, camera_file):
        # 初始化数据序列
        self.data_path = data_path
        self.rgbs_list, self.deps_list = [], []
        self.cam_info = None

        # 读取数据序列
        if not tl.files.folder_exists(self.data_path):
            print("[skip] data_path is not found: {}".format(self.data_path))
            pass
        rgb_path = os.path.join(self.data_path, color_folder)
        dep_path = os.path.join(self.data_path, depth_folder)
        if tl.files.folder_exists(rgb_path):
            self.rgbs_list = tl.files.load_file_list(rgb_path, regx='\.jpg')
            self.rgbs_list = [rgb_path + '/' + rgb for rgb in self.rgbs_list]
        else:
            print("[skip] rgb_path is not found: {}".format(self.data_path))
        if tl.files.folder_exists(dep_path):
            self.deps_list = tl.files.load_file_list(dep_path, regx='\.png')
            self.deps_list = [dep_path + '/' + dep for dep in self.deps_list]
        else:
            print("[skip] dep_path is not found: {}".format(self.data_path))

        if os.path.exists(camera_file):
            params = [float(line.split(" : ")[1]) for line in open(camera_file,"r")]
            cam_K = np.array([[params[0], 0.0, params[2]],
                            [0.0, params[1], params[3]],
                            [0.0, 0.0, 1.0]])
            cam_distCoef = np.array(params[4:])
            self.cam_info = dict(zip(['K', 'distCoef'], [cam_K, cam_distCoef]))
        else:
            print("[skip] cam_file is not found: {}".format(self.data_path))
        print("Total number of own images found:", len(self.rgbs_list))

    def inference_data(self):
        time0 = time.time()
        self.coords_uv_list, self.coords_xyz_list = list(), list()
        for _, (rgb_name, dep_name) in enumerate(zip(self.rgbs_list, self.deps_list)):
            input_2d, init_h, init_w = measure(lambda: read_2dfiles(rgb_name, dep_name, self.size_2d), 'read_2dfiles')
            humans, _, _ = measure(lambda: self.e_2d.inference(input_2d), 'e_2d.inference')
            print('got %d humans from %s' % (len(humans), rgb_name[:-4]))

            if len(humans) is 0:
                self.coords_uv_list.append(None)
                self.coords_xyz_list.append(None)
            else:
                coords2d, coords2d_conf, coords2d_vis = tranform_keypoints2d(humans[0].body_parts, init_w, init_h, 0.1)
                input_3d, trafo_params = measure(lambda: read_3dfiles(dep_name, self.cam_info, coords2d, coords2d_vis, self.size_3d), 'read_3dfiles')

                # 1.热图
                coords3d, coords3d_conf = measure(lambda: self.e_3d.inference(input_3d), 'e_3d.inference')
                coords3d_pred = coords3d * trafo_params['scale'] + trafo_params['root']
                coords3d_pred_proj = Camera(self.cam_info['K'], self.cam_info['distCoef']).unproject(coords2d, coords3d_pred[:, -1])
                cond = coords2d_conf > coords3d_conf  # use backproj only when 2d was visible and 2d/3d roughly matches
                coords3d_pred[cond, :] = coords3d_pred_proj[cond, :]
                coords3d_conf[cond] = coords2d_conf[cond]

                # 2.积分回归
                # coords3d = measure(lambda: e_3d.regression(input_3d), 'e_3d.inference')
                # coords3d_pred = coords3d * trafo_params['scale'] + trafo_params['root']

                self.coords_uv_list.append(coords2d)
                self.coords_xyz_list.append(coords3d_pred*1000.0)
               
        mean = (time.time() - time0) / len(self.rgbs_list)
        print('inference all took: %f, mean: %f, FPS: %f' % (time.time() - time0, mean, 1.0 / mean))

    def write_kp_to_txt(self, filename2d, filename3d):
        def coords_to_txt(coords_list, filename):
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

        coords_to_txt(self.coords_uv_list, os.path.join(self.data_path, filename2d))
        coords_to_txt(self.coords_xyz_list, os.path.join(self.data_path, filename3d))

    def write_kp_to_mat(self, filename):
        for idx, coords in enumerate(self.coords_uv_list):
            if coords is None:
                self.coords_uv_list[idx] = np.array([np.array([50000,50000,50000])] * 18)
            else:
                coords_z = 50000 * np.ones_like(coords[:,0])
                self.coords_uv_list[idx] = np.concatenate((coords, np.expand_dims(coords_z,-1)), axis=1)
        
        for idx, coords in enumerate(self.coords_xyz_list):
            if coords is None:
                self.coords_xyz_list[idx] = np.array([np.array([50000,50000,50000])] * 18)

        import scipy.io as sio
        sio.savemat(os.path.join(self.data_path, filename), {'keypoints2d':self.coords_uv_list,'keypoints3d':self.coords_xyz_list})


if __name__ == '__main__':
    base_npz_path = 'models/2d/hao28-pose-average.npz' # str, default='', help='path to npz', required=True
    base_model_name = 'hao28_experimental' # str, default='hao28_experimental', help='mobilenet | mobilenet2 | hao28_experimental'
    head_npz_path = 'models/3d/voxelposenet-False.npz' # str, default='', help='path to npz', required=True
    head_model_name = 'voxelposenet' # str, default='voxelposenet', help='voxelposenet | pixelposenet'
    kp_extractor = KeypointExtractor(base_model_name, base_npz_path, head_model_name, head_npz_path)
    
    state = 'completed'
    while True:
        # TODO：获取管道消息
    
        # 使用state表示当前获取到的状态
        if state is 'completed': # 数据获取结束
            # 指定或解析出以下信息
            data_path = 'g:/Lab/dataset/Gait-Database/2019-03-25_22-45'
            color_folder = 'color'
            depth_folder = 'depth/filter'
            camera_file = 'g:/Lab/dataset/Gait-Database/rgbIP/kinect2.txt'
            # 获取文件
            kp_extractor.get_files(data_path, color_folder, depth_folder, camera_file)
            # 使用网络推断数据
            kp_extractor.inference_data()
            # 存储数据
            kp_extractor.write_kp_to_txt('keypoints2d.txt', 'keypoints3d.txt')
            kp_extractor.write_kp_to_mat('keypoints.mat')
            break