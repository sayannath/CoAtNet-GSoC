import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


def conv_3x3(input_layer, num_filters: int = 64, downsample: bool = False):
    """
    3x3 Convolutional Stem Stage.

    Args:
        input_layer: input tensor
        num_filters: number of filters in the convolutional layer
        downsample: whether to downsample the input layer or not

    Returns:
        output tensor
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


def mbconv_block(
    input_layer,
    input_channel: int,
    output_channel: int,
    strides: int = 1,
    expansion_rate: int = 4,
    se_ratio: float = 0.25,
    dropout_rate: float = 0.1,
):
    """
    Mobile Inverted Block - Stage 2

    Refer: https://github.com/keras-team/keras-applications/blob/06fbeb0f16e1304f239b2296578d1c50b15a983a/keras_applications/efficientnet.py#L119

    Args:
        input_layer: input tensor
        input_channel (int): number of input channel
        output_channel (int): number of output channel
        strides (int): number of strides
        expansion (int): expansion rate
        se_ratio (float): between 0 and 1, fraction to squeeze the input filters.
        dropout_rate (float): between 0 and 1, fraction of the input units to drop.

    Returns:
        output tensor for the block
    """

    conv_1 = layers.Conv2D(
        filters=input_channel * expansion_rate,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
    )(input_layer)

    bn_1 = layers.BatchNormalization()(conv_1)
    act_1 = tf.nn.gelu(bn_1)

    dconv_1 = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(strides, strides),
        padding="same",
        use_bias=False,
    )(act_1)
    bn_2 = layers.BatchNormalization()(dconv_1)
    act_2 = tf.nn.gelu(bn_2)

    # SE Module
    filters_se = max(1, int(input_channel * se_ratio))
    se = layers.GlobalAveragePooling2D()(act_2)

    se = layers.Reshape((1, 1, input_channel * expansion_rate))(se)

    se = layers.Conv2D(
        filters=filters_se,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
    )(se)

    act_3 = tf.nn.gelu(se)

    se = layers.Conv2D(
        filters=input_channel * expansion_rate,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation="sigmoid",
        padding="same",
        use_bias=False,
    )(act_3)

    mul = layers.multiply([act_2, se])

    conv_2 = layers.Conv2D(
        filters=output_channel,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
    )(mul)

    bn_3 = layers.BatchNormalization()(conv_2)
    drop_1 = layers.Dropout(rate=dropout_rate)(bn_3)
    return drop_1
