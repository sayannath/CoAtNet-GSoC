import tensorflow as tf
from configs.model_configs import get_model_config
from tensorflow import keras

from coatnet.models.blocks import conv_3x3, mbconv_block


def get_training_model(
    model_name: str,
    image_shape: tuple,
    expansion_rate: int,
    se_ratio: int,
    num_classes: int,
) -> keras.Model:
    """
    Implements CoAtNet family of models given a configuration.
    References:
        (1) https://arxiv.org/pdf/2106.04803.pdf
    """
    configs = get_model_config(model_name)

    input_layer = keras.Input(shape=image_shape)  # Input Layer
    conv_1 = conv_3x3(
        input_layer=input_layer,
        num_filters=configs.out_channels[0],
        downsample=True,
    )  # Conv Stem Block
    conv_2 = conv_3x3(input_layer=conv_1, num_filters=configs.out_channels[0])
    print(conv_2.shape)

    mbconv_1 = mbconv_block(
        input_layer=conv_2,
        input_channel=configs.out_channels[0],
        output_channel=configs.out_channels[1],
        strides=2,
        expansion_rate=expansion_rate,
        se_ratio=se_ratio,
        dropout_rate=0.1,
    )
    print(mbconv_1.shape)

    mbconv_2 = mbconv_block(
        input_layer=mbconv_1,
        input_channel=configs.out_channels[1],
        output_channel=configs.out_channels[2],
        strides=2,
        expansion_rate=expansion_rate,
        se_ratio=se_ratio,
        dropout_rate=0.1,
    )
    print(mbconv_2.shape)

    model = keras.models.Model(input_layer, mbconv_2)
    return model
