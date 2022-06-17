import tensorflow as tf
from tensorflow.keras import layers


def conv_3x3(input_layer, num_filters: int = 64, downsample: bool = False):
    """
    3x3 Convolutional Stem Stage.

    Args:
        input_layer: input tensor
        num_filters: number of filters in the convolutional layer
        downsample: whether to downsample the input layer or not
    """
    stride = 1 if downsample == False else 2
    conv_1 = layers.Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        use_bias=False,
    )(input_layer)
    bn_1 = layers.BatchNormalization()(conv_1)
    act_1 = tf.nn.gelu(bn_1)
    return act_1
