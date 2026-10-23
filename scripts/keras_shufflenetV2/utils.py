#-*- coding:utf-8 -*-
#'''
# Created on 18-8-14 下午4:39
#
# @Author: Greg Gao(laygin)
#'''
import os
import keras
import tensorflow as tf
# from keras import backend as K
# from keras_applications.imagenet_utils import _obtain_input_shape
# from keras.models import Model
# from keras.engine.topology import get_source_inputs
# from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
# from keras.layers import GlobalAveragePooling2D,Input, Dense
# from keras.layers import MaxPool2D,AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
import numpy as np


def channel_split(x, name=''):
    # equipartition
    # in_channels = x.shape[-1]
    # ip = in_channels // 2
    # c_hat = keras.layers.Lambda(lambda z: z[:, :, :, 0:ip], name='%s_sp%d_slice' % (name, 0))(x)
    # c = keras.layers.Lambda(lambda z: z[:, :, :, ip:], name='%s_sp%d_slice' % (name, 1))(x)
    in_channels = keras.ops.shape(x)[-1]
    ip = in_channels // 2
    c_hat = x[:, :, :, :ip]
    c = x[:, :, :, ip:]
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = tf.keras.backend.reshape(x, [-1, height, width, 2, channels_per_split])
    x = tf.keras.backend.permute_dimensions(x, (0,1,2,4,3))
    x = tf.keras.backend.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):
    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = f'stage{stage}_block{block}'
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, f'{prefix}_spl')
        inputs = c

    x = keras.layers.Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name=f'{prefix}_1x1conv_1')(inputs)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=f'{prefix}_bn_1x1conv_1')(x)
    x = keras.layers.Activation('relu', name=f'{prefix}_relu_1x1conv_1')(x)
    x = keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name=f'{prefix}_3x3dwconv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=f'{prefix}_bn_3x3dwconv')(x)
    x = keras.layers.Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name=f'{prefix}_1x1conv_2')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=f'{prefix}_bn_1x1conv_2')(x)
    x = keras.layers.Activation('relu', name=f'{prefix}_relu_1x1conv_2')(x)

    if strides < 2:
        ret = keras.layers.Concatenate(axis=bn_axis, name=f'{prefix}_concat_1')([x, c_hat])
    else:
        s2 = keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name=f'{prefix}_3x3dwconv_2')(inputs)
        s2 = keras.layers.BatchNormalization(axis=bn_axis, name=f'{prefix}_bn_3x3dwconv_2')(s2)
        s2 = keras.layers.Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name=f'{prefix}_1x1_conv_3')(s2)
        s2 = keras.layers.BatchNormalization(axis=bn_axis, name=f'{prefix}_bn_1x1conv_3')(s2)
        s2 = keras.layers.Activation('relu', name=f'{prefix}_relu_1x1conv_3')(s2)
        ret = keras.layers.Concatenate(axis=bn_axis, name=f'{prefix}_concat_2')([x, s2])

    ret = keras.layers.Lambda(channel_shuffle, name=f'{prefix}_channel_shuffle')(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

    return x
