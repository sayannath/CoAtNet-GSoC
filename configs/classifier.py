import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.model_name = "coatnet_0"
    config.image_height = 224  # Image height
    config.image_width = 224  # Image width
    config.image_channels = 3  # Number of channels

    config.expansion_rate = 4  # Expansion rate of inverted bottleneck
    config.se_ratio = 0.25  # Squeeze and Excitation ratio
    config.batch_size = 128  # Batch size for training.
    config.epochs = 100  # Number of epochs to train for.
    config.num_classes = 1000  # Number of classes in the dataset.

    return config
