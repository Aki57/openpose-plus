import math
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf

flag = 1

loaded1 = np.load('models/pre-trained/backbone/mobilenet.npz')['params']
print(len(loaded1))

loaded2 = np.load('models/pose100000.npz')
print(len(loaded2.files))

for i in range(100, 136):
    print('loaded1[{}] shape = {} type = {}'.format(i, loaded1[i].shape, type(loaded1[i])))

    name = loaded2.files[i]
    shape = loaded2[name].shape
    print('loaded2[{}] shape = {} name = {} type = {}'.format(i, shape, name, type(loaded2[name])))

    if flag:
        continue

    for d1 in range(shape[0]):
        if(len(shape)==1):
            print('d1 = {} val1 = {} val2 = {}'.format(d1, loaded1[i][d1], loaded2[name][d1]))
            continue
        else:
            print('d1 = {}'.format(d1))
        for d2 in range(shape[1]):
            print('--d2 = {}'.format(d2))
            for d3 in range(shape[2]):
                print('----d3 = {}'.format(d3))
                for d4 in range(shape[3]):
                    print('------d4 = {} val1 = {} val2 = {}'.format(d4, loaded1[i][d1][d2][d3][d4], loaded2[name][d1][d2][d3][d4])) 


save_list_names = loaded2.files
save_list_vars = []
for i in range(len(save_list_names)):
    if i < 105:
        print('loaded1[{}] shape = {} type = {}'.format(i, loaded1[i].shape, type(loaded1[i])))
        save_list_vars.append(loaded1[i])
    else:
        name = save_list_names[i]
        shape = loaded2[name].shape
        print('loaded2[{}] shape = {} name = {} type = {}'.format(i, shape, name, type(loaded2[name])))
        save_list_vars.append(loaded2[name])
    print('save_list_vars[{}] = {}\n'.format(i, save_list_vars[i]))

save_var_dict = {save_list_names[idx]: val for idx, val in enumerate(save_list_vars)}
np.savez('models/pose-trans.npz', **save_var_dict)