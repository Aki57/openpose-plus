import math
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf

flag = 3

loaded1 = np.load('models/voxelposenet1000.npz')
print(len(loaded1.files))
loaded2 = np.load('models/voxelposenet4000.npz')
print(len(loaded2.files))

for i, name in enumerate(loaded2.files):
    shape1 = loaded1[name].shape
    shape2 = loaded2[name].shape
    print('loaded1[{}] shape = {} name = {} type = {}'.format(i, shape1, name, type(loaded1[name])))
    print('loaded2[{}] shape = {} name = {} type = {}'.format(i, shape2, name, type(loaded2[name])))

    if flag==0:
        continue
    elif flag==1:
        print('val1 = {}'.format(loaded1[name]))
        print('val2 = {}'.format(loaded2[name]))
    else:
        for d1 in range(shape1[0]):
            if(len(shape1)==1):
                print('d1 = {} val1 = {} val2 = {}'.format(d1, loaded1[name][d1], loaded2[name][d1]))
                continue
            else:
                print('d1 = {}'.format(d1))
            for d2 in range(shape1[1]):
                print('--d2 = {}'.format(d2))
                for d3 in range(shape1[2]):
                    print('----d3 = {}'.format(d3))
                    if flag==2:
                        for d4 in range(shape1[3]):
                            print('------d4 = {} val1 = {} val2 = {}'.format(d4, loaded1[name][d1][d2][d3][d4], loaded2[name][d1][d2][d3][d4])) 
                    else:
                        for d4 in range(shape1[3]):
                            print('------d4 = {}'.format(d4))
                            for d5 in range(shape1[4]):
                                print('--------d5 = {} val1 = {} val2 = {}'.format(d5, loaded1[name][d1][d2][d3][d4][d5], loaded2[name][d1][d2][d3][d4][d5])) 
