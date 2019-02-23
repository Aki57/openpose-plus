import math
import os
import cv2
import string
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf

loaded1 = np.load('models/pre-trained/voxelposenet-40000-False.npz')
loaded2 = np.load('models/voxelposenet.npz')
ori_list_names = loaded1.files
save_list_names = loaded2.files
print(len(ori_list_names))
print(len(save_list_names))

rename_dict = {'W_conv3d':'weights', 'W_deconv3d':'weights', 'b_conv3d':'biases', 'b_deconv3d':'biases'}

save_list_vars = []
for i, name2 in enumerate(save_list_names):
    for key in rename_dict.keys():
        if key in name2:
            name1 = name2.replace(key, rename_dict[key])[:-2]

    print('loaded1[{}] shape = {} name = {} type = {}'.format(i, loaded1[name1].shape, name1, type(loaded1[name1])))
    print('loaded2[{}] shape = {} name = {} type = {}'.format(i, loaded2[name2].shape, name2, type(loaded2[name2])))
    # print('val1 = {}'.format(loaded1[name1]))
    # print('val2 = {}'.format(loaded2[name2]))

    save_list_vars.append(loaded1[name1])
    print('save_list_vars[{}] = {}\n'.format(i, save_list_vars[i].shape))

save_var_dict = {save_list_names[idx]: val for idx, val in enumerate(save_list_vars)}
np.savez('models/voxelposenet.npz', **save_var_dict)