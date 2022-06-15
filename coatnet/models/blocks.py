import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


def conv_3x3(input_layer, num_filters=64, downsample=False):
    stride = 1 if downsample == False else 2
    conv_1 = layers.Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        use_bias=False,
    )(input_layer)
    bn_1 = layers.BatchNormalization()(conv_1)
    act_1 = tfa.layers.GELU()(bn_1)
    return act_1
