import tensorflow as tf

from train_config import config
from ..inference.common import rename_tensor

__all__ = [
    'get_base_model',
    'get_head_model',
    'get_model',
]


def _input_image(height, width, n_channel=3, data_format='channels_last', name='image'):
    """Create a placeholder for input image."""
    # TODO: maybe make it a Layer in tensorlayer
    if data_format == 'channels_last':
        shape = (None, height, width, n_channel)
    elif data_format == 'channels_first':
        shape = (None, n_channel, height, width)
    else:
        raise ValueError('invalid data_format: %s' % data_format)
    return tf.placeholder(tf.float32, shape, name)


def _input_voxel(width, height, depth, data_format='channels_last', name='voxel'):
    """Create a placeholder for input image."""
    if data_format == 'channels_last':
        shape = (1, width, height, depth, 19)
    elif data_format == 'channels_first':
        shape = (1, 19, width, height, depth)
    else:
        raise ValueError('invalid data_format: %s' % data_format)
    return tf.placeholder(tf.float32, shape, name)


def get_model(name):
    if name == 'hao28_experimental':
        from .models_hao28_experimental import model
    elif name == 'mobilenet':
        from .models_mobilenet import model
    elif name == 'mobilenet2':
        from .models_mobilenet2 import model
    elif name == 'voxelposenet':
        from .models_head_voxelposenet import model
    elif name == 'pixelposenet':
        from .models_head_pixelposenet import model
    else:
        raise RuntimeError('unknown head model %s' % name)
    return model
model = get_model(config.MODEL.name)


def get_base_model(base_model_name):

    def model_func(target_size, data_format):
        base_model = get_model(base_model_name)
        n_pos = 19
        image = _input_image(target_size[0], target_size[1], target_size[2], data_format)
        _, b1_list, b2_list, _ = base_model(image, n_pos, False, False, data_format=data_format)
        conf_tensor = b1_list[-1].outputs
        pafs_tensor = b2_list[-1].outputs
        with tf.variable_scope('outputs'):
            return image, rename_tensor(conf_tensor, 'conf'), rename_tensor(pafs_tensor, 'paf')

    return model_func


def get_head_model(head_model_name):

    def model_func(target_size, use_slim, data_format):
        head_model = get_model(head_model_name)
        voxel = _input_voxel(target_size[0], target_size[1], target_size[2], data_format)
        _, voxel_out = head_model(voxel, n_pos=18, reuse=False, use_slim=use_slim, data_format=data_format)
        voxel_tensor = voxel_out.outputs
        with tf.variable_scope('outputs'):
            return voxel, rename_tensor(voxel_tensor, 'score')

    return model_func
