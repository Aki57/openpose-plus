from enum import Enum
import time
from distutils.dir_util import mkpath

import cv2
import numpy as np
import tensorflow as tf

from openpose_plus.utils import PoseInfo, create_voxelgrid, get_3d_heatmap, get_kp_heatmap

regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu

# actually the output order of Openpose
class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

CocoPairs = [(1, 2), (2, 3), (3, 4),  # right arm
             (1, 8), (8, 9), (9, 10),  # right leg
             (1, 5), (5, 6), (6, 7),  # left arm
             (1, 11), (11, 12), (12, 13),  # left leg
             (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]  # = 17
CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def read_depth(dep_path):
    """smooth depth image by matlab interpret."""
    if '.mat' in dep_path:
        from scipy.io import loadmat
        dep_img = loadmat(dep_path)['depthim_incolor']
    else:
        from scipy.misc import imread
        dep_img = imread(dep_path)
    return dep_img


def read_2dfiles(rgb_path, dep_path, height, width, data_format='channels_last'):
    """Read image file and resize to network input size."""
    dep_img = read_depth(dep_path) / 20.0
    init_h, init_w = dep_img.shape

    rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_img.shape[:2] != dep_img.shape:
        rgb_img = cv2.resize(rgb_img, (init_w, init_h))

    input_2d = np.concatenate((rgb_img, np.expand_dims(dep_img, -1)), -1)
    if height is not None and width is not None:
        input_2d = cv2.resize(input_2d, (width, height))
    if data_format == 'channels_first':
        input_2d = input_2d.transpose([2, 0, 1])

    return input_2d / 255.0, init_h, init_w


def read_3dfiles(dep_path, cam_info, coords2d, coordsvis, width, height, depth, data_format='channels_last'):
    """Read image file and resize to network input size."""
    dep_img = read_depth(dep_path) / 1000.0
    voxel_grid, voxel_coords2d, voxel_coordsvis, trafo_params = create_voxelgrid(cam_info, dep_img, coords2d, (width, height, depth), 1.2, coordsvis)
    voxel_kp, _ = get_kp_heatmap(voxel_coords2d, (width, height), 3.0, voxel_coordsvis)
    voxel_kp = np.tile(np.expand_dims(voxel_kp, 2), [1, 1, depth, 1])
    voxel_grid = np.expand_dims(voxel_grid, -1)
    input_3d = np.concatenate((voxel_grid, voxel_kp), 3)
    return np.expand_dims(input_3d, 0), trafo_params


def tranform_keypoints2d(body, width, height, kp_score_thresh=0.25):
    coords2d = np.zeros((18, 2))
    coords2d_conf = np.zeros((18))
    coords2d_vis = coords2d_conf > 0.0
    for i in body.keys():
        coords2d_conf[i] = body[i].score
        coords2d[i,0] = body[i].x * width
        coords2d[i,1] = body[i].y * height
        if coords2d_conf[i] > kp_score_thresh:
            coords2d_vis[i] = True
    return coords2d, coords2d_conf, coords2d_vis


class Profiler(object):

    def __init__(self):
        self.count = dict()
        self.total = dict()

    def __del__(self):
        if self.count:
            self.report()

    def report(self):
        sorted_costs = sorted([(t, name) for name, t in self.total.items()])
        sorted_costs.reverse()
        names = [name for _, name in sorted_costs]
        hr = '-' * 80
        print(hr)
        print('%-12s %-12s %-12s %s' % ('tot (s)', 'count', 'mean (ms)', 'name'))
        print(hr)
        for name in names:
            tot, cnt = self.total[name], self.count[name]
            mean = tot / cnt
            print('%-12f %-12d %-12f %s' % (tot, cnt, mean * 1000, name))

    def __call__(self, name, duration):
        if name in self.count:
            self.count[name] += 1
            self.total[name] += duration
        else:
            self.count[name] = 1
            self.total[name] = duration


_default_profiler = Profiler()


def measure(f, name=None):
    if not name:
        name = f.__name__
    t0 = time.time()
    result = f()
    duration = time.time() - t0
    _default_profiler(name, duration)
    return result


def draw_humans(npimg, humans):
    npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:
        # draw point
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairs):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return npimg


def do_plot():
    import matplotlib.pyplot as plt
    plt.show()


def plot_humans(image, heatMat, pafMat, humans, name):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    a = fig.add_subplot(2, 3, 1)

    plt.imshow(draw_humans(image, humans))

    a = fig.add_subplot(2, 3, 2)
    tmp = np.amax(heatMat[:, :, :-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 3, 4)
    a.set_title('Vectormap-x')
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 3, 5)
    a.set_title('Vectormap-y')
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    mkpath('vis')
    plt.savefig('vis/result-%s.png' % name)


def plot_human2d(rgb_path, dep_path, coords2d, idx=None, coordsvis=None):
    import matplotlib.pyplot as plt
    if coordsvis is None:
        coordsvis = np.ones_like(coords2d[:,0], dtype=np.bool)

    dep_img = read_depth(dep_path)
    rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_img.shape[:2] != dep_img.shape:
        rgb_img = cv2.resize(rgb_img, (dep_img.shape[1], dep_img.shape[0]))

    # 2d display for 3d body projecttion on Kinect frame
    plt.figure(figsize=[8, 7])
    plt.subplots_adjust(0, 0.05, 1, 0.95, 0, 0)
    plt.subplot(2,1,1)
    plt.title('2D Body on Kinect-Color ({})'.format(idx))
    plt.imshow(rgb_img)
    plt.plot(coords2d[:,0], coords2d[:,1], '.')
    for pair in CocoPairs:
        if coordsvis[pair[0]] and coordsvis[pair[1]]:
            plt.plot(coords2d[pair,0], coords2d[pair,1], linewidth=2)
        else:
            plt.plot(coords2d[pair,0], coords2d[pair,1], color='k')
    plt.draw()
    
    plt.subplot(2,1,2)
    plt.title('2D Body on Kinect-Depth ({})'.format(idx))
    plt.imshow(dep_img)
    plt.plot(coords2d[:,0], coords2d[:,1], 'r.')
    for pair in CocoPairs:
        if coordsvis[pair[0]] and coordsvis[pair[1]]:
            plt.plot(coords2d[pair,0], coords2d[pair,1], linewidth=2)
        else:
            plt.plot(coords2d[pair,0], coords2d[pair,1], color='k')
    plt.draw()


def plot_human3d(rgb_path, dep_path, coords3d, cam, idx=0, coordsvis=None):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if coordsvis is None:
        coords3d = np.array(coords3d)/100.0
        coordsvis = np.ones_like(coords3d[:,0], dtype=np.bool)

    dep_img = read_depth(dep_path)
    rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_img.shape[:2] != dep_img.shape:
        rgb_img = cv2.resize(rgb_img, (dep_img.shape[1], dep_img.shape[0]))
    coords2d_proj = cam.project(coords3d[coordsvis,:])

    # 2d display for 3d body projecttion on Kinect frame
    plt.figure(figsize=[15, 8])
    plt.subplots_adjust(0, 0.05, 1, 0.95, 0, 0)
    gs = gridspec.GridSpec(2,2)
    plt.subplot(gs[0,0])
    plt.title('3D Body Projection on Kinect-Color ({})'.format(idx))
    plt.imshow(rgb_img)
    plt.plot(coords2d_proj[:,0], coords2d_proj[:,1], '.')
    plt.draw()
    
    plt.subplot(gs[1,0])
    plt.title('3D Body Projection on Kinect-Depth ({})'.format(idx))
    plt.imshow(dep_img)
    plt.plot(coords2d_proj[:,0], coords2d_proj[:,1], 'r.')
    plt.draw()

    # 3d display for body joints
    ax = plt.subplot(gs[:,1], projection='3d')
    ax.scatter(coords3d[:,0], coords3d[:,2], coords3d[:,1], 'g')  # 绘制数据点
    for pair in CocoPairs:
        if coordsvis[pair[0]] and coordsvis[pair[1]]:
            ax.plot(coords3d[pair,0], coords3d[pair,2], coords3d[pair,1], linewidth=2)
        else:
            ax.plot(coords3d[pair,0], coords3d[pair,2], coords3d[pair,1], color='k')

    # stable range of sight
    axis_range = np.array([[-1.1,1.1,-1.1], [1.1,-1.1,1.1]])
    if coordsvis[1] == True: # Check if root keypoint is visible
        axis_range = np.array([[-1.1,1.8,-1.1], [1.1,-0.4,1.1]])
        axis_range += coords3d[1,:]
    elif coordsvis[8] == True: # if not try R-hip
        axis_range += coords3d[8,:]
    elif coordsvis[11] == True: # if not try L-hip
        axis_range += coords3d[11,:]
    else:
        axis_range += np.mean(coords3d, axis=0)
    ax.set_xlim3d(axis_range[:,0])
    ax.set_ylim3d(axis_range[:,2])
    ax.set_zlim3d(axis_range[:,1])
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Height')  # 坐标轴
    plt.draw()


def rename_tensor(x, name):
    # FIXME: use tf.identity(x, name=name) doesn't work
    new_shape = []
    for d in x.shape:
        try:
            d = int(d)
        except:
            d = -1
        new_shape.append(d)
    return tf.reshape(x, new_shape, name=name)
