import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
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


class MbConv_Block(keras.layers.Layer):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        se_ratio: float,
        strides: int,
        conv_shortcut: bool,
        expansion: int,
        dropout_rate: float,
        **kwargs
    ):
        super(MbConv_Block, self).__init__(**kwargs)

        self.bn = keras.layers.BatchNormalization()
        self.act = tfa.layers.GELU()

        if conv_shortcut:
            if strides > 1:
                self.conv_shortcut_pool = keras.layers.AveragePooling2D(
                    pool_size=(strides, strides), padding="same"
                )
            self.conv_shortcut_layer = keras.layers.Conv2D(
                output_channel, kernel_size=1, strides=1, padding="same", use_bias=False
            )

        self.conv_1 = (
            keras.layers.Conv2D(
                filters=input_channel * expansion,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
                use_bias=False,
            ),
        )
        self.bn_1 = keras.layers.BatchNormalization()
        self.act_1 = tfa.layers.GELU()
        self.dconv_1 = (
            keras.layers.DepthwiseConv2D(
                kernel_size=(3, 3),
                strides=(strides, strides),
                padding="same",
                use_bias=False,
            ),
        )
        self.bn_2 = keras.layers.BatchNormalization()
        self.act_2 = tfa.layers.GELU()

        self.se_module = SE_Module(
            input_channel * expansion, ratio=se_ratio / expansion, activation="gelu"
        )

        self.conv_2 = keras.layers.Conv2D(
            filters=output_channel,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=False,
        )
        self.drop_1 = keras.layers.Dropout(rate=dropout_rate)

    def __make_divisible(self, value, divisor, min_value=None):
        """Refer: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py"""
        if min_value is None:
            min_value = divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def call(self, inputs):
        x = self.bn(inputs)
        x = self.act(x)
        pre_act = x

        if hasattr(self, "conv_shortcut_layer"):
            if hasattr(self, "conv_shortcut_pool"):
                pre_act = self.conv_shortcut_pool(pre_act)
            shortcut = self.conv_shortcut_layer(pre_act)
        else:
            shortcut = inputs

        x = self.conv_1[0](x)
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.dconv_1[0](x)
        x = self.bn_2(x)
        x = self.act_2(x)
        x = self.se_module(x)
        x = self.conv_2(x)
        x = self.drop_1(x)
        x = keras.layers.Add()([x, shortcut])
        return x


class SE_Module(keras.layers.Layer):
    def __init__(
        self,
        num_filters: int,
        ratio: int = 16,
        divisor: int = 8,
        activation="gelu",
        channel_format="channels_last",
        use_bias: bool = True,
        **kwargs
    ):
        super(SE_Module, self).__init__(**kwargs)
        self.channel_axis = -1
        self.h_axis, self.w_axis = [1, 2]

        if channel_format == "channels_first":
            self.channel_axis = 1
            self.h_axis, self.w_axis = [2, 3]

        reduction = self.__make_divisible(num_filters * ratio, divisor)
        self.conv_1 = keras.layers.Conv2D(
            reduction, kernel_size=1, strides=1, padding="same", use_bias=use_bias
        )

        self.act_1 = tfa.layers.GELU()

        if activation == "relu":
            self.act_1 = keras.layers.ReLU()

        self.conv_2 = keras.layers.Conv2D(
            num_filters, kernel_size=1, strides=1, padding="same", use_bias=use_bias
        )

        self.act_2 = keras.layers.Activation("sigmoid")
        self.multiply = keras.layers.Multiply()

    def __make_divisible(self, value, divisor, min_value=None):
        """Refer: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py"""
        if min_value is None:
            min_value = divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def call(self, inputs):
        x = tf.reduce_mean(inputs, [self.h_axis, self.w_axis], keepdims=True)
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.multiply([inputs, x])
        return x
