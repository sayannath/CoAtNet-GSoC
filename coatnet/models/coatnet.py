import tensorflow as tf
from configs.model_configs import get_model_config
from tensorflow import keras

from coatnet.models.blocks import conv_3x3


class CoAtNet(keras.models.Model):
    def __init__(self, model_name: str, image_shape: tuple, num_classes: int, **kwargs):
        super(CoAtNet, self).__init__(**kwargs)
        self.model_name = model_name
        self.image_shape = image_shape
        self.num_classes = num_classes

    def get_training_model(self):
        """
        Implements CoAtNet family of models given a configuration.
        References:
            (1) https://arxiv.org/pdf/2106.04803.pdf
        """
        configs = get_model_config(self.model_name)

        input_layer = keras.Input(shape=self.image_shape)  # Input Layer
        conv_1 = conv_3x3(
            input_layer=input_layer,
            num_filters=configs.out_channels[0],
            downsample=True,
        )  # Conv Stem Block
        conv_2 = conv_3x3(input_layer=conv_1, num_filters=configs.out_channels[0])
        print(conv_2.shape)
