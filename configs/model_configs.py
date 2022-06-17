"""
Configuratiionns for different CoAtNet variants. 
Referred from: https://arxiv.org/abs/2106.04803
"""


import ml_collections


def coatnet_0_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.num_blocks = [2, 2, 3, 5, 2]  # L
    configs.out_channels = [64, 96, 192, 384, 768]  # D
    return configs


def coatnet_1_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.num_blocks = [2, 2, 6, 14, 2]  # L
    configs.out_channels = [64, 96, 192, 384, 768]  # D
    return configs


def coatnet_2_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.num_blocks = [2, 2, 6, 14, 2]  # L
    configs.out_channels = [128, 128, 256, 512, 1024]  # D
    return configs


def coatnet_3_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.num_blocks = [2, 2, 6, 14, 2]  # L
    configs.out_channels = [192, 192, 384, 768, 1536]  # D
    return configs


def coatnet_4_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.num_blocks = [2, 2, 12, 28, 2]  # L
    configs.out_channels = [192, 192, 384, 768, 1536]  # D
    return configs


def coatnet_5_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.num_blocks = [2, 2, 12, 28, 2]  # L
    configs.out_channels = [192, 256, 512, 1280, 2048]  # D
    return configs


def coatnet_6_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.num_blocks = [2, 2, 6, 8, 42, 2]  # L
    configs.out_channels = [192, 192, 384, 768, 1536, 2048]  # D
    return configs


def coatnet_7_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.num_blocks = [2, 2, 6, 8, 42, 2]  # L
    configs.out_channels = [192, 256, 512, 1024, 2048, 3072]  # D
    return configs


def get_model_config(model_name: str) -> ml_collections.ConfigDict:
    """
    Get the model configuration for the given model name.

    Args:
        model_name (str): Name of the model.

    Returns:
        ml_collections.ConfigDict: Model configuration.
    """
    if model_name == "coatnet_0":
        return coatnet_0_config()
    elif model_name == "coatnet_1":
        return coatnet_1_config()
    elif model_name == "coatnet_2":
        return coatnet_2_config()
    elif model_name == "coatnet_3":
        return coatnet_3_config()
    elif model_name == "coatnet_4":
        return coatnet_4_config()
    elif model_name == "coatnet_5":
        return coatnet_5_config()
    elif model_name == "coatnet_6":
        return coatnet_6_config()
    else:
        return coatnet_7_config()
