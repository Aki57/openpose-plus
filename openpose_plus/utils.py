import math
import os
from distutils.dir_util import mkpath

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.misc as smc
import tensorflow as tf
from train_config import config

if config.TRAIN.train_stage == '2d':
    n_pos = config.MODEL.n_pos
    hout = config.MODEL.hout
    wout = config.MODEL.wout

## read cmu data
class CmuMeta:
    """ Be used in PoseInfo. 
        Neck = 0
        Nose = 1
        BodyCenter = 2
        LShoulder = 3
        LElbow = 4
        LWrist = 5
        LHip = 6
        LKnee = 7
        LAnkle = 8
        RShoulder = 9
        RElbow = 10
        RWrist = 11
        RHip = 12
        RKnee = 13
        RAnkle = 14
        LEye = 15
        LEar = 16
        REye = 17
        REar = 18
    """

    def __init__(self, rgb_url, depth_url, annos2d, annos3d, scores, min_count=5, min_score=0.5):
        self.rgb_url = rgb_url
        self.depth_url = depth_url
        self.joint2d_list = []
        self.joint3d_list = []
        
        # 对CMU数据集的转换
        transform = [1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 17, 15, 18, 16]

        for bodyidx in range(len(scores)):
            p2d = annos2d[bodyidx]
            p3d = annos3d[bodyidx]
            score = scores[bodyidx]

            new_joints2d = []
            new_joints3d = []

            valid_count = 0

            for idx in transform:
                if score[idx] >= min_score and int(p2d[idx][0]) in range(0, 1920) and int(p2d[idx][1]) in range(0, 1080):
                    valid_count = valid_count + 1
                    new_joints2d.append(p2d[idx])
                    new_joints3d.append(p3d[idx])
                else:
                    new_joints2d.append(np.array([-1000, -1000]))
                    new_joints3d.append(np.array([0, 0, -1000]))

            # for background
            new_joints2d.append(np.array([-1000, -1000]))
            new_joints3d.append(np.array([0, 0, -1000]))

            # min threshold for joint count
            if valid_count < min_count:
                continue

            self.joint2d_list.append(new_joints2d)
            self.joint3d_list.append(new_joints3d)


class PoseInfo:
    """ Use CMU for 3d pose estimation """

    def __init__(self, data_base_dir, metas_filename, min_count, min_score):
        self.base_dir = data_base_dir
        self.metas_path = os.path.join(self.base_dir, metas_filename)
        self.metas = []
        self.min_count = min_count
        self.min_score = min_score

        if not os.path.exists(self.metas_path):
            print("[skip] meta.mat is not found: {}".format(self.base_dir))
        else:
            self.get_image_annos()

    def get_image_annos(self):
        """
        Read Meta file, and get and check the image list.
        Skip missing images.
        """
        cam = sio.loadmat(self.metas_path)['cam'][0]
        cam_K = cam['K'][0]
        cam_distCoef = cam['distCoef'][0]
        self.cam_calib = dict(zip(['K', 'distCoef'], [cam_K, cam_distCoef]))

        rgb_names = sio.loadmat(self.metas_path)['rgb'][0]
        depth_names = sio.loadmat(self.metas_path)['depth'][0]
        anno_names = sio.loadmat(self.metas_path)['anno'][0]
        for idx in range(len(anno_names)):
            rgb_path = os.path.join(self.base_dir, rgb_names[idx][0])
            depth_path = os.path.join(self.base_dir, depth_names[idx][0])
            anno_path = os.path.join(self.base_dir, anno_names[idx][0])
            # filter that some images might not in the list
            if not os.path.exists(rgb_path):
                print("[skip] rgb is not found: {}".format(rgb_path))
                continue
            if not os.path.exists(depth_path):
                print("[skip] depth is not found: {}".format(depth_path))
                continue
            if not os.path.exists(anno_path):
                print("[skip] anno is not found: {}".format(anno_path))
                continue
            
            annos2d = sio.loadmat(anno_path)['joints2d'][0]
            annos3d = sio.loadmat(anno_path)['joints3d'][0]
            scores = sio.loadmat(anno_path)['scores'][0]

            meta = CmuMeta(rgb_path, depth_path, annos2d, annos3d, scores, self.min_count, self.min_score)
            if len(meta.joint2d_list) != 0:
                self.metas.append(meta)

        print("Overall get {} valid pose images from {}".format(len(self.metas), self.base_dir))

    def get_2d_data_list(self):
        rgb_list, dep_list, joint2d_list = [], [], []
        for meta in self.metas:
            rgb_list.append(meta.rgb_url)
            dep_list.append(meta.depth_url)
            joint2d_list.append(meta.joint2d_list)
        return rgb_list, dep_list, joint2d_list

    def get_3d_data_list(self):
        cam_list, dep_list, joint2d_list, joint3d_list = [], [], [], []
        for meta in self.metas:
            for _, (joints2d, joints3d) in enumerate(zip(meta.joint2d_list, meta.joint3d_list)):
                cam_list.append(self.cam_calib)
                dep_list.append(meta.depth_url)
                joint2d_list.append(joints2d)
                joint3d_list.append(joints3d)
        return cam_list, dep_list, joint2d_list, joint3d_list

    def get_val_data_list(self):
        cam_list, rgb_list, dep_list, joint3d_list = [], [], [], []
        for meta in self.metas:
            cam_list.append(self.cam_calib)
            rgb_list.append(meta.rgb_url)
            dep_list.append(meta.depth_url)
            joint3d_list.append(meta.joint3d_list)
        return cam_list, rgb_list, dep_list, joint3d_list


class Camera:
    def __init__(self, K, distCoef, use_distort=True):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.D = np.squeeze(distCoef)[:5]
        self.D_apply = use_distort

    def project(self, xyz_coords):
        """ Projects a (x, y, z) tuple of world coords into the image frame. """
        # scaling by z coords
        xyz_coords = np.reshape(xyz_coords, [-1, 3])
        xy_coords = self._from_hom(xyz_coords)

        # distortion
        if self.D_apply:
            x, y = xy_coords[:,0], xy_coords[:,1]
            r2 = x*x + y*y
            d = self.D
            deltaR = 1 + d[0]*r2 + d[1]*r2*r2 + d[4]*r2*r2*r2
            deltaX = 2*d[2]*x*y + d[3]*(r2 + 2*x*x)
            deltaY = 2*d[3]*x*y + d[2]*(r2 + 2*y*y)
            xy_coords[:,0] = deltaR * x + deltaX
            xy_coords[:,1] = deltaR * y + deltaY

        # normalized points and mapping
        xy_coords_h = self._to_hom(xy_coords)
        uv_coords = np.matmul(xy_coords_h, np.transpose(self.K, [1, 0]))
        return self._from_hom(uv_coords)

    def unproject(self, uv_coords, z_coords):
        """ Projects a (x, y, z) tuple of world coords into the world frame. """
        # normalized points and mapping
        uv_coords_h = self._to_hom(np.reshape(uv_coords, [-1, 2]))
        xy_coords_h = np.matmul(uv_coords_h, np.transpose(self.K_inv, [1, 0]))

        # undistortion iterations
        if self.D_apply:
            x0, y0 = xy_coords_h[:,0], xy_coords_h[:,1]
            x, y = x0, y0
            d = self.D
            for i in range(5):
                r2 = x*x + y*y
                deltaR = 1 / (1 + d[0]*r2 + d[1]*r2*r2 + d[4]*r2*r2*r2)
                deltaX = 2*d[2]*x*y + d[3]*(r2 + 2*x*x)
                deltaY = 2*d[3]*x*y + d[2]*(r2 + 2*y*y)
                x = (x0 - deltaX)*deltaR
                y = (y0 - deltaY)*deltaR
            xy_coords_h[:,0] = x
            xy_coords_h[:,1] = y

        # scaling by z coords
        z_coords = np.reshape(z_coords, [-1, 1])
        xyz_coords = z_coords * xy_coords_h
        return xyz_coords

    @staticmethod
    def _to_hom(coords):
        """ Turns the [N, D] coord matrix into homogeneous coordinates [N, D+1]. """
        coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], 1)
        return coords_h

    @staticmethod
    def _from_hom(coords_h):
        """ Turns the homogeneous coordinates [N, D+1] into [N, D]. """
        coords = coords_h[:, :-1] / (coords_h[:, -1:] + 1e-10)
        return coords


def read_depth(dep_path):
    """Load depth image of diff type."""
    if '.mat' in dep_path:
        dep_img = sio.loadmat(dep_path)['depthim_incolor']
    elif '.png' in dep_path:
        dep_img = smc.imread(dep_path)
    else:
        raise Exception('Unknown depth format')
    return dep_img


def aug_depth(depth_map):
    """Add block randomly in depth image."""
    height, width = depth_map.shape
    max_depth = np.amax(depth_map)

    if np.random.uniform() > 0.75:
        w0, w1 = np.random.randint(width/4, size=2)
        h0, h1 = np.random.randint(height/2, size=2)
        depth = np.random.uniform(0.5, 1.0)*max_depth

        mask1 = np.zeros_like(depth_map, dtype=np.bool)
        mask1[h0:height-h1, w0:width-w1] = True
        mask2 = depth_map > depth
        mask = np.logical_and(mask1, mask2)
        depth_map[mask] = depth

    if np.random.uniform() > 0.5:
        rand_w, rand_h = np.random.randint(width/3, size=2)
        w0 = np.random.randint(width-rand_w)
        h0 = np.random.randint(height-rand_h)
        depth = np.random.uniform(0.25, 0.75)*max_depth

        mask1 = np.zeros_like(depth_map, dtype=np.bool)
        mask1[h0:h0+rand_h, w0:w0+rand_w] = True
        mask2 = depth_map > depth
        mask = np.logical_and(mask1, mask2)
        depth_map[mask] = depth

    # TODO：增加高斯噪声

    return depth_map


def _get_depth_value(map, coord2d, crop_size=25):
    """ Extracts a depth value from a map.
        Checks for the closest value in a given neighborhood. """
    expand = False
    if len(map.shape) == 2:
        map = np.expand_dims(map, 2)
        expand = True
    assert len(map.shape) == 3, "Map has to be of Dimension 2 or 3."
    shape = map.shape
        
    crop_size = np.round(crop_size).astype('int')
    while True:
        # make sure crop size cant exceed image dims
        crop_size = shape[0] if shape[0] <= crop_size else crop_size
        crop_size = shape[1] if shape[1] <= crop_size else crop_size

        # work out the coords to actually lie in the crop
        min_c = np.round(coord2d - crop_size // 2).astype('int')
        max_c = min_c + crop_size

        # check if we left map
        for dim in [0, 1]:
            left_value, right_value, right_range = min_c[dim], max_c[dim], shape[1-dim]
            shift = left_value if left_value<0 else (right_value-right_range if right_value>right_range else 0)
            min_c[dim], max_c[dim] = (left_value-shift, right_value-shift)

        # perform crop
        map_c = map[min_c[1]:max_c[1], min_c[0]:max_c[0], :]
        map_c = np.squeeze(map_c) if expand else map_c

        # find valid depths
        X, Y = np.where(np.not_equal(map_c, 0.0))
        if X.shape[0] == 0:
            crop_size *= 2  # if not successful use larger crop
        else:
            break

    # calculate distance
    grid = np.stack([X, Y], 1) - (coord2d - min_c)
    dist = np.sqrt(np.sum(np.square(grid), 1))

    # find element with minimal distance
    nn_ind = np.argmin(dist)
    z_val = map_c[X[nn_ind], Y[nn_ind]] 
    return z_val

def _voxelize(cam, depth_warped, mask, voxel_root, grid_size, grid_size_m, f=1.0):
    """ Creates a voxelgrid from given input. """
    grid_size = np.reshape(grid_size, [1, 3]).astype('int32')

    # 1. Vectorize depth and project into world
    H, W = np.meshgrid(range(0, depth_warped.shape[0]), range(0, depth_warped.shape[1]), indexing='ij')
    h_vec = np.reshape(H[mask], [-1])
    w_vec = np.reshape(W[mask], [-1])
    uv_vec = np.stack([w_vec, h_vec], 1)
    z_vec = np.reshape(depth_warped[mask], [-1])
    pcl_xyz = cam.unproject(uv_vec, z_vec)

    # 2. Scale down to voxel size and quantize
    pcl_xyz_rel = pcl_xyz - voxel_root
    pcl_xyz_01 = (pcl_xyz_rel - grid_size_m[0, :]) / (grid_size_m[1, :] - grid_size_m[0, :])
    pcl_xyz_vox = pcl_xyz_01 * grid_size

    # 3. Discard unnecessary parts of the pointcloud
    pcl_xyz_vox = pcl_xyz_vox.astype('int32')
    cond_x = np.logical_and(pcl_xyz_vox[:, 0] < grid_size[0, 0], pcl_xyz_vox[:, 0] >= 0)
    cond_y = np.logical_and(pcl_xyz_vox[:, 1] < grid_size[0, 1], pcl_xyz_vox[:, 1] >= 0)
    cond_z = np.logical_and(pcl_xyz_vox[:, 2] < grid_size[0, 2], pcl_xyz_vox[:, 2] >= 0)
    cond = np.logical_and(cond_x, np.logical_and(cond_y, cond_z))
    pcl_xyz_vox = pcl_xyz_vox[cond, :]

    # 4. Set values in the grid
    voxel_grid = np.zeros((grid_size[0, :]))
    voxel_grid[pcl_xyz_vox[:, 0],
                pcl_xyz_vox[:, 1],
                pcl_xyz_vox[:, 2]] = 1.0

    # 5. Trafo params
    voxel_root += grid_size_m[0, :]
    voxel_scale = (grid_size_m[1, :] - grid_size_m[0, :]) / grid_size
    trafo_params = {'root':voxel_root, 'scale':voxel_scale}

    return voxel_grid, trafo_params

def create_voxelgrid(cam, depth_warped, coords2d, grid_size, f=1.0, coordsvis=None):
    """ Creates a voxelgrid from given input. """
    cam = Camera(cam['K'], cam['distCoef'])
    mask = np.logical_not(depth_warped == 0.0)
    if coordsvis is None:
        coordsvis = (coords2d != np.array([-1000, -1000]))[:,0]
    
    grid_size_m = np.array([[-1.1, -1.1, -1.1], [1.1, 1.1, 1.1]]) #symmetric grid

    if coordsvis[1] == True: # Check if root keypoint is visible
        coord2d_root = coords2d[1,:]
        grid_size_m = np.array([[-1.1, -0.4, -1.1], [1.1, 1.8, 1.1]]) #asymmetric grid
    elif coordsvis[8] == True: # if not try R-hip
        coord2d_root = coords2d[8,:]
    elif coordsvis[11] == True: # if not try L-hip
        coord2d_root = coords2d[11,:]
    else:
        coord2d_root = np.mean(coords2d, axis=0)

    # find approx depth for root
    z_value = _get_depth_value(depth_warped, coord2d_root, 25)
    if z_value == 0.0:
        print("Could not extract depth value. Skipping sample.")
        return None

    grid_size_m *= f # TODO：焦距究竟多少

    grid_size = np.reshape(grid_size, [1, 3]).astype('int32')
    root_xyz = cam.unproject(coord2d_root, z_value)  # neck world coordinates

    # get a voxel located at the root_xyz
    voxel_grid, trafo_params = _voxelize(cam, depth_warped, mask, root_xyz, grid_size, grid_size_m, f)

    # calculate pseudo 2D coordinates at neck depth in voxel coords
    pseudo_coord3d = cam.unproject(coords2d, z_value)  # project all points onto the neck depth
    pseudo_coord3d = (pseudo_coord3d - trafo_params['root']) / trafo_params['scale']
    voxel_coords2d = pseudo_coord3d[:, :2]

    return voxel_grid, voxel_coords2d, coordsvis, trafo_params


def get_kp_heatmap(coords_uv, output_size, sigma, coords_vis=None):
    """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
        with variance sigma for multiple coordinates."""
    assert len(output_size) == 2, "Output size has to be of Dimension 2."

    if coords_vis is None:
        coords_vis = np.ones_like(coords_uv[:, 0], dtype=np.float32)
        coords_vis = np.greater(coords_vis, 0.5)

    cond_1_in = np.logical_and(np.less(coords_uv[:, 0], output_size[0]-1), np.greater(coords_uv[:, 0], 0))
    cond_2_in = np.logical_and(np.less(coords_uv[:, 1], output_size[1]-1), np.greater(coords_uv[:, 1], 0))
    cond = np.logical_and(coords_vis, np.logical_and(cond_1_in, cond_2_in))

    # create meshgrid
    X, Y = np.meshgrid(range(0, output_size[0]), range(0, output_size[1]), indexing='ij')
    X = np.tile(np.expand_dims(X, -1), [1, 1, coords_uv.shape[0]])
    Y = np.tile(np.expand_dims(Y, -1), [1, 1, coords_uv.shape[0]])

    # compute gaussian
    X_b = np.array(X - coords_uv[:, 0]).astype('float32')
    Y_b = np.array(Y - coords_uv[:, 1]).astype('float32')
    dist = np.square(X_b) + np.square(Y_b)
    heatmap_kp = np.exp(- dist / np.square(sigma)) * cond

    return heatmap_kp, cond


def get_3d_heatmap(coords_xyz, output_size, sigma, coords_vis=None):
    assert len(output_size) == 3

    if coords_vis is None:
        coords_vis = np.ones_like(coords_xyz[:, 0], dtype=np.float32)
        coords_vis = np.greater(coords_vis, 0.5)

    cond_1_in = np.logical_and(np.less(coords_xyz[:, 0], output_size[0]-1), np.greater(coords_xyz[:, 0], 0))
    cond_2_in = np.logical_and(np.less(coords_xyz[:, 1], output_size[1]-1), np.greater(coords_xyz[:, 1], 0))
    cond_3_in = np.logical_and(np.less(coords_xyz[:, 2], output_size[2]-1), np.greater(coords_xyz[:, 2], 0))
    cond = np.logical_and(np.logical_and(coords_vis, cond_1_in), np.logical_and(cond_2_in, cond_3_in))

    # create meshgrid
    X, Y, Z = np.meshgrid(range(0, output_size[0]), range(0, output_size[1]), range(0, output_size[2]), indexing='ij')
    X = np.tile(np.expand_dims(X, -1), [1, 1, 1, coords_xyz.shape[0]])
    Y = np.tile(np.expand_dims(Y, -1), [1, 1, 1, coords_xyz.shape[0]])
    Z = np.tile(np.expand_dims(Z, -1), [1, 1, 1, coords_xyz.shape[0]])

    # compute gaussian 
    X_b = np.array(X - coords_xyz[:, 0]).astype('float32')
    Y_b = np.array(Y - coords_xyz[:, 1]).astype('float32')
    Z_b = np.array(Z - coords_xyz[:, 2]).astype('float32')
    dist = np.square(X_b) + np.square(Y_b) + np.square(Z_b)
    heatmap_3d = np.exp(- dist / np.square(sigma)) * cond

    return heatmap_3d, cond


def keypoints_affine(coords, transform_matrix):
    """Transform keypoint coordinates according to a given affine transform matrix.
    OpenCV format, x is width.

    Parameters
    -----------
    coords: list of tuple/list the coordinates
        e.g., the keypoint coordinates of one person in an image.
    transform_matrix : numpy.array
        Transform matrix, OpenCV format.
    """
    coords = np.asarray(coords)
    coords = coords.transpose([1, 0])
    
    if coords.shape[0] == 2: 
        coords = np.insert(coords, 2, 1, axis=0)
        coords_result = np.matmul(transform_matrix, coords)
        coords_result = coords_result[0:2, :]
    else:
        ori_coord_z = coords[2,:].copy()
        coords[2,:] = 1 
        coords_result = np.matmul(transform_matrix, coords)
        coords_result[2,:] = ori_coord_z

    coords_result = coords_result.transpose([1, 0])
    return coords_result


def keypoint_flip(coords, output_size, axis=0, coords_vis=None, flip_list=(0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16)):
    new_coords = []
    new_coords_vis = []

    for k in flip_list:
        point = coords[k]
        point[axis] = output_size[axis] - point[axis]

        cond = True if coords_vis is None else coords_vis[k]
        for i in range(len(output_size)):
            cond = cond and int(point[i]) in range(0, output_size[i])

        new_coords_vis.append(cond)
        new_coords.append(point if cond else [-output_size[0]]*len(output_size))

    return new_coords, new_coords_vis


def get_heatmap(annos, height, width):
    # 19 for coco, 15 for MPII
    num_joints = 19

    # the heatmap for every joints takes the maximum over all people
    joints_heatmap = np.zeros((num_joints, height, width), dtype=np.float32)

    # among all people
    for joint in annos:
        # generate heatmap for every keypoints
        # loop through all people and keep the maximum

        for i, points in enumerate(joint):
            if points[0] < 0 or points[1] < 0:
                continue
            joints_heatmap = put_heatmap(joints_heatmap, i, points, 8.0)

    # 0: joint index, 1:y, 2:x
    joints_heatmap = joints_heatmap.transpose((1, 2, 0))

    # background
    joints_heatmap[:, :, -1] = np.clip(1 - np.amax(joints_heatmap, axis=2), 0.0, 1.0)

    mapholder = []
    for i in range(0, 19):
        a = cv2.resize(np.array(joints_heatmap[:, :, i]), (hout, wout))
        mapholder.append(a)
    mapholder = np.array(mapholder)
    joints_heatmap = mapholder.transpose(1, 2, 0)

    return joints_heatmap.astype(np.float16)


def put_heatmap(heatmap, plane_idx, center, sigma):
    center_x, center_y = center
    _, height, width = heatmap.shape[:3]

    th = 4.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))

    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

    exp_factor = 1 / 2.0 / sigma / sigma

    ## fast - vectorize
    arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]
    y_vec = (np.arange(y0, y1 + 1) - center_y)**2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x)**2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
    return heatmap


def get_vectormap(annos, height, width):
    """

    Parameters
    -----------


    Returns
    --------


    """
    num_joints = 19

    limb = list(
        zip([2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16],
            [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]))

    vectormap = np.zeros((num_joints * 2, height, width), dtype=np.float32)
    counter = np.zeros((num_joints, height, width), dtype=np.int16)

    for joint in annos:
        if len(joint) != 19:
            print('THE LENGTH IS NOT 19 ERROR:', len(joint))
        for i, (a, b) in enumerate(limb):
            a -= 1
            b -= 1

            v_start = joint[a]
            v_end = joint[b]
            # exclude invisible or unmarked point
            if v_start[0] < -100 or v_start[1] < -100 or v_end[0] < -100 or v_end[1] < -100:
                continue
            vectormap = cal_vectormap(vectormap, counter, i, v_start, v_end)

    vectormap = vectormap.transpose((1, 2, 0))
    # normalize the PAF (otherwise longer limb gives stronger absolute strength)
    nonzero_vector = np.nonzero(counter)

    for i, y, x in zip(nonzero_vector[0], nonzero_vector[1], nonzero_vector[2]):

        if counter[i][y][x] <= 0:
            continue
        vectormap[y][x][i * 2 + 0] /= counter[i][y][x]
        vectormap[y][x][i * 2 + 1] /= counter[i][y][x]

    mapholder = []
    for i in range(0, n_pos * 2):
        a = cv2.resize(np.array(vectormap[:, :, i]), (hout, wout), interpolation=cv2.INTER_AREA)
        mapholder.append(a)
    mapholder = np.array(mapholder)
    vectormap = mapholder.transpose(1, 2, 0)

    return vectormap.astype(np.float16)


def cal_vectormap(vectormap, countmap, i, v_start, v_end):
    """

    Parameters
    -----------


    Returns
    --------


    """
    _, height, width = vectormap.shape[:3]

    threshold = 8
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]
    length = math.sqrt(vector_x**2 + vector_y**2)
    if length == 0:
        return vectormap

    min_x = max(0, int(min(v_start[0], v_end[0]) - threshold))
    min_y = max(0, int(min(v_start[1], v_end[1]) - threshold))

    max_x = min(width, int(max(v_start[0], v_end[0]) + threshold))
    max_y = min(height, int(max(v_start[1], v_end[1]) + threshold))

    norm_x = vector_x / length
    norm_y = vector_y / length

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - v_start[0]
            bec_y = y - v_start[1]
            dist = abs(bec_x * norm_y - bec_y * norm_x)

            # orthogonal distance is < then threshold
            if dist > threshold:
                continue
            countmap[i][y][x] += 1
            vectormap[i * 2 + 0][y][x] = norm_x
            vectormap[i * 2 + 1][y][x] = norm_y

    return vectormap


def fast_vectormap(vectormap, countmap, i, v_start, v_end):
    """

    Parameters
    -----------


    Returns
    --------


    """
    _, height, width = vectormap.shape[:3]
    _, height, width = vectormap.shape[:3]

    threshold = 8
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]

    length = math.sqrt(vector_x**2 + vector_y**2)
    if length == 0:
        return vectormap

    min_x = max(0, int(min(v_start[0], v_end[0]) - threshold))
    min_y = max(0, int(min(v_start[1], v_end[1]) - threshold))

    max_x = min(width, int(max(v_start[0], v_end[0]) + threshold))
    max_y = min(height, int(max(v_start[1], v_end[1]) + threshold))

    norm_x = vector_x / length
    norm_y = vector_y / length

    x_vec = (np.arange(min_x, max_x) - v_start[0]) * norm_y
    y_vec = (np.arange(min_y, max_y) - v_start[1]) * norm_x

    xv, yv = np.meshgrid(x_vec, y_vec)

    dist_matrix = abs(xv - yv)
    filter_matrix = np.where(dist_matrix > threshold, 0, 1)
    countmap[i, min_y:max_y, min_x:max_x] += filter_matrix
    for y in range(max_y - min_y):
        for x in range(max_x - min_x):
            if filter_matrix[y, x] != 0:
                vectormap[i * 2 + 0, min_y + y, min_x + x] = norm_x
                vectormap[i * 2 + 1, min_y + y, min_x + x] = norm_y
    return vectormap


def draw_results(images, heats_ground, heats_result, pafs_ground, pafs_result, masks, name=''):
    """Save results for debugging.

    Parameters
    -----------
    images : a list of RGB images
    heats_ground : a list of keypoint heat maps or None
    heats_result : a list of keypoint heat maps or None
    pafs_ground : a list of paf vector maps or None
    pafs_result : a list of paf vector maps or None
    masks : a list of mask for people
    """
    for i in range(len(images)):
        if heats_ground is not None:
            heat_ground = heats_ground[i]
        if heats_result is not None:
            heat_result = heats_result[i]
        if pafs_ground is not None:
            paf_ground = pafs_ground[i]
        if pafs_result is not None:
            paf_result = pafs_result[i]
        if masks is not None:
            mask = masks[i, :, :, 0]
            mask = mask[:, :, np.newaxis]
            mask1 = np.repeat(mask, n_pos, 2)
            mask2 = np.repeat(mask, n_pos * 2, 2)
        image = images[i]

        fig = plt.figure(figsize=(8, 8))
        a = fig.add_subplot(2, 3, 1)
        plt.imshow(image)

        if pafs_ground is not None:
            a = fig.add_subplot(2, 3, 2)
            a.set_title('Vectormap_ground')
            if masks is not None:
                vectormap = paf_ground * mask2
            else:
                vectormap = paf_ground
            tmp2 = vectormap.transpose((2, 0, 1))
            tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
            plt.imshow(tmp2_odd, alpha=0.3)
            plt.colorbar()
            plt.imshow(tmp2_even, alpha=0.3)

        if pafs_result is not None:
            a = fig.add_subplot(2, 3, 3)
            a.set_title('Vectormap result')
            if masks is not None:
                vectormap = paf_result * mask2
            else:
                vectormap = paf_result
            tmp2 = vectormap.transpose((2, 0, 1))
            tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
            plt.imshow(tmp2_odd, alpha=0.3)

            plt.colorbar()
            plt.imshow(tmp2_even, alpha=0.3)

        if heats_result is not None:
            a = fig.add_subplot(2, 3, 4)
            a.set_title('Heatmap result')
            if masks is not None:
                heatmap = heat_result * mask1
            else:
                heatmap = heat_result
            tmp = heatmap
            tmp = np.amax(heatmap[:, :, :-1], axis=2)

            plt.colorbar()
            plt.imshow(tmp, alpha=0.3)

        if heats_ground is not None:
            a = fig.add_subplot(2, 3, 5)
            a.set_title('Heatmap ground truth')
            if masks is not None:
                heatmap = heat_ground * mask1
            else:
                heatmap = heat_ground
            tmp = heatmap
            tmp = np.amax(heatmap[:, :, :-1], axis=2)

            plt.colorbar()
            plt.imshow(tmp, alpha=0.3)

        if masks is not None:
            a = fig.add_subplot(2, 3, 6)
            a.set_title('Mask')
            plt.colorbar()
            plt.imshow(mask[:, :, 0], alpha=0.3)

        mkpath(config.LOG.vis_path)
        plt.savefig(os.path.join(config.LOG.vis_path, '%s%d.png' % (name, i)), dpi=300)


def vis_annos(image, annos, name=''):
    """Save results for debugging.

    Parameters
    -----------
    images : single RGB image
    annos  : annotation, list of lists
    """

    fig = plt.figure(figsize=(8, 8))
    a = fig.add_subplot(1, 1, 1)

    plt.imshow(image)
    for people in annos:
        for idx, jo in enumerate(people):
            if jo[0] > 0 and jo[1] > 0:
                plt.plot(jo[0], jo[1], '*')

    mkpath(config.LOG.vis_path)
    plt.savefig(os.path.join(config.LOG.vis_path, 'keypoints%s.png' % (name)), dpi=300)


def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """

    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
    repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)

    return repeated_tesnor