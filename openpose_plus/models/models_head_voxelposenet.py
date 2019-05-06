# refer to VoxelPoseNet
import math
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (ConcatLayer, Conv3dLayer, DeConv3dLayer, InputLayer, MaxPool3d, ElementwiseLayer, BatchNormLayer)
__all__ = [
    'model',
]

W_init = tf.contrib.layers.xavier_initializer()
b_init = tf.constant_initializer(0.0001)
decay = 0.999

def model(x, n_pos, reuse=False, use_slim=False, data_format='channels_last'):
    """ Init deconv3d kernal value. """
    def get_deconv_init(shape):
        width = shape[0]
        height = shape[1]
        depth = shape[2]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        trilinear = np.zeros([width, height, depth])
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    value = (1 - abs(x / f - c)) * (1 - abs(y / f - c)) * (1 - abs(z / f - c))
                    trilinear[x, y, z] = value
        weights = np.zeros(shape)
        for i in range(shape[3]):
            weights[:, :, :, i, i] = trilinear

        return np.array(weights, dtype=np.float32)

    """ Infer normalized depth coordinate. """
    with tf.variable_scope('VoxelPoseNet', reuse):
        scorevolume_list = list()
        # input
        init_shape = x.get_shape().as_list()
        x = InputLayer(x, name='input')

        # ENCODER
        conv_per_block = [1, 1, 2] if use_slim else [2, 2, 2]
        in_chan_per_block = [init_shape[4], 64, 128]
        out_chan_per_block = [64, 128, 256]

        skip_list = list()
        for block_id, (layer_num, in_chan, out_chan) in enumerate(zip(conv_per_block, in_chan_per_block, out_chan_per_block), 1):
            last_x = None
            for layer_id in range(layer_num):
                if last_x!=None:
                    x = ConcatLayer([last_x, x], 4)
                    in_chan += out_chan
                last_x = x if layer_num>1 else None

                x = Conv3dLayer(x,
                    shape=(3,3,3,in_chan,out_chan),
                    strides=(1,1,1,1,1),
                    act=lambda x : tf.nn.leaky_relu(x, 0.01),
                    W_init=W_init,
                    b_init=b_init,
                    name='conv%d_%d'%(block_id, layer_id+1))

            x = MaxPool3d(x, filter_size=(2, 2, 2), strides=(2, 2, 2), name='pool%d' % block_id)
            skip_list.append(x)

        # DECODER: Use skip connections to get the details right
        skip_list = skip_list[::-1]
        scorevolume_list = []
        for block_id, skip_x in enumerate(skip_list, 1):
            skip_last_dim = skip_x.outputs.get_shape().as_list()[4]
            # upconv to next layer and incorporate the skip connection
            if block_id < len(skip_list):
                next_x = skip_list[block_id]
                next_shape = next_x.outputs.get_shape().as_list()

                up_x = DeConv3dLayer(skip_x,
                    shape=(4,4,4,skip_last_dim,skip_last_dim),
                    output_shape=(next_shape[0],next_shape[1],next_shape[2],next_shape[3],skip_last_dim),
                    strides=(1,2,2,2,1),
                    W_init=tf.constant_initializer(value=get_deconv_init([4,4,4,skip_last_dim,skip_last_dim])),
                    name='upconv%d' % block_id)

                next_x = ConcatLayer([up_x, next_x], 4)
                next_x = Conv3dLayer(next_x, 
                    shape=(3,3,3,skip_last_dim+next_shape[4],next_shape[4]),
                    strides=(1,1,1,1,1),
                    act=lambda x : tf.nn.leaky_relu(x, 0.01),
                    W_init=W_init,
                    b_init=b_init,
                    name='conv_decoder%d' % block_id)
                skip_list[block_id] = next_x

            k = 2**(5 - block_id)
            s = 2**(4 - block_id)

            if block_id < len(skip_list):
                skip_x = Conv3dLayer(skip_x,
                    shape=(1,1,1,skip_last_dim,n_pos),
                    strides=(1,1,1,1,1),
                    W_init=W_init,
                    b_init=b_init,
                    name='det%d'%block_id)
                
                scorevolume = DeConv3dLayer(skip_x,
                    shape=(k,k,k,n_pos,n_pos),
                    output_shape=(init_shape[0],init_shape[1],init_shape[2],init_shape[3],n_pos),
                    strides=(1,s,s,s,1),
                    W_init=tf.constant_initializer(value=get_deconv_init([k,k,k,n_pos,n_pos])),
                    name='scorevolume%d' % block_id)
            else:
                # Final estimation
                skip_x = DeConv3dLayer(skip_x,
                    shape=(k,k,k,skip_last_dim,skip_last_dim),
                    output_shape=(init_shape[0],init_shape[1],init_shape[2],init_shape[3],skip_last_dim),
                    strides=(1,s,s,s,1),
                    W_init=tf.constant_initializer(value=get_deconv_init([k,k,k,skip_last_dim,skip_last_dim])),
                    name='upconv_final')

                scorevolume = Conv3dLayer(skip_x,
                    shape=(1,1,1,skip_last_dim,n_pos),
                    strides=(1,1,1,1,1),
                    W_init=W_init,
                    b_init=b_init,
                    name='scorevolume_final')

            if len(scorevolume_list) != 0:
                scorevolume = ElementwiseLayer([scorevolume, scorevolume_list[-1]], combine_fn=tf.add, name='add')
            # scorevolume = BatchNormLayer(scorevolume, is_train=bool(1-reuse), name='bn%d'%block_id)
            scorevolume_list.append(scorevolume)

        return scorevolume_list, scorevolume_list[-1]