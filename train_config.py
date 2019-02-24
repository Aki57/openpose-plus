import os

from easydict import EasyDict as edict

config = edict()

config.TRAIN = edict()
config.TRAIN.train_stage = '3d'  # 2d, 3d stage
config.TRAIN.train_mode = 'single'  # single, parallel
config.TRAIN.node_num = 2  # indicate num of parallel gpus or cpus
config.TRAIN.n_epoch = 10
config.TRAIN.save_interval = 5000 # 5000
config.TRAIN.log_interval = 1

config.MODEL = edict()
config.MODEL.model_path = 'models'  # save directory

if config.TRAIN.train_stage == '2d':
    config.TRAIN.batch_size = 8
    config.TRAIN.n_step = 1  # total number of step: 600000
    config.TRAIN.lr_init = 4e-5  # initial learning rate
    config.TRAIN.lr_decay_interval = 136106  # evey number of step to decay lr
    config.TRAIN.lr_decay_factor = 0.333  # decay lr factor
    config.TRAIN.weight_decay_factor = 5e-4

    config.MODEL.n_pos = 19  # number of keypoints + 1 for background
    config.MODEL.hin = 368  # input size during training , 240
    config.MODEL.win = 368
    config.MODEL.hout = int(config.MODEL.hin / 8)  # output size during training (default 46)
    config.MODEL.wout = int(config.MODEL.win / 8)
    config.MODEL.name = 'hao28_experimental'  # vgg, vggtiny, mobilenet, hao28_experimental
    if (config.MODEL.hin % 16 != 0) or (config.MODEL.win % 16 != 0):
        raise Exception("image size should be divided by 16")

elif config.TRAIN.train_stage == '3d':
    config.TRAIN.batch_size = 2
    config.TRAIN.n_step = 40000  # total number of step: 40000
    config.TRAIN.lr_init = 1e-4  # initial learning rate
    config.TRAIN.lr_decay_interval = 10000  # evey number of step to decay lr
    config.TRAIN.lr_decay_factor = 0.1  # decay lr factor
    config.TRAIN.weight_decay_factor = 5e-4 # TODO:l2 norm
    
    config.MODEL.n_pos = 18  # number of keypoints + 1 for background
    config.MODEL.xdim = 64
    config.MODEL.ydim = 64
    config.MODEL.zdim = 64
    config.MODEL.name = 'voxelposenet'  # voxelposenet, pixelposenet
    config.MODEL.use_slim = False

else:
    raise Exception('Unknown model stage')

config.DATA = edict()
config.DATA.train_data = 'custom'  # coco, custom, coco_and_custom
config.DATA.coco_version = '2017'  # MSCOCO version 2014 or 2017
config.DATA.data_path = 'data/cmu_dataset'
config.DATA.your_images_path = os.path.join('data', 'your_data', 'images')
config.DATA.your_annos_path = os.path.join('data', 'your_data', 'coco.json')

config.LOG = edict()
config.LOG.vis_path = 'vis'

# config.VALID = edict()

# import json
# def log_config(filename, cfg):
#     with open(filename, 'w') as f:
#         f.write("================================================\n")
#         f.write(json.dumps(cfg, indent=4))
#         f.write("\n================================================\n")
