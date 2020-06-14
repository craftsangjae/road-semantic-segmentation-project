"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""


class ModelConfig:
    # Model Overall Setting에 대한 설정
    INPUT_SHAPE = (None, None, 3)
    NUM_CLASSES = 8
    SEPARATE_BG_CLASSIFIER = False # BackGround를 구분하는 네트워크 따로 구분
    SEPARATE_CRACK_CLASSIFIER = True # Crack을 구분하는 네트워크 따로 구분
    USE_GROUPNORM = True # Batch Normalization을 쓸지, Group normalization을 쓸지에 대한 코드
    GROUPS = 16 # Group Normalization에 관련된 Hyper Parameter

    # BackBone Network에 대한 설정
    BACKBONE = "MobileNetV2"
    """
    Choose one, [0.35, 0.5, 0.75, 1.0, 1.3, 1.4]
    """
    BACKBONE_ALPHA = 1.4

    """
    | Feature Map Name    | OS |
    | ----------------    | -- |
    | block_1_expand_relu |  2 |
    | block_3_expand_relu |  4 |
    | block_6_expand_relu |  8 |
    | block_13_expand_relu| 16 |
    | block_16_project_BN | 32 |
    """
    BACKBONE_LOW_FEATURE_MAP_NAME = 'block_3_expand_relu'
    BACKBONE_HIGH_FEATURE_MAP_NAME = "block_13_expand_relu"
    BACKBONE_FREEZE = True # Freezing Backbone or Not

    # ASPP Network에 대한 설정
    ASPP_NUM_FEATURES = 256 # ASPP Module 내에서의
    ASPP_ATROUS_RATE = (6, 12, 18)

    # Decoder Network에 대한 설정
    DECODER_NUM_DEPTH = 2
    DECODER_NUM_FEATURES = 256
    DECODER_NUM_SKIP_FEATURES = 48
    DECODER_USE_SEPARABLE_CONV = False

    def __init__(self, config_dict=None):
        if config_dict is not None:
            if 'ModelConfig' in config_dict:
                config_dict = config_dict['ModelConfig']
            for key, value in config_dict.items():
                self.__setattr__(key, value)

    def to_dict(self):
        """ Convert Configuration Attributes to dict Object
        :return:
        """
        return {name: getattr(self, name)
                for name in dir(self) if name.isupper()}
