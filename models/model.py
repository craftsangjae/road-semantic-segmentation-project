"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from .layers import MeanShift, aspp_module, deeplab_decoder, ResizeLike
from .config import ModelConfig


def DeepLabV3(config: ModelConfig):
    """ Build DeepLabv3+ model

    :param config: ModelConfiguration Class,
                   Reference - `models.config` module
    :return: Keras Model
    """
    # 전처리 부분 구성하기
    inputs = Input(shape=config.INPUT_SHAPE)

    # BackBone Network 구성하기
    if config.BACKBONE == 'MobileNetV2':
        from tensorflow.python.keras.applications import MobileNetV2
        preprocess = MeanShift()(inputs)  # (0,255) -> (-1, 1)
        base = MobileNetV2(input_tensor=preprocess,
                           alpha=config.BACKBONE_ALPHA,
                           include_top=False)
        skip_input = (base
                      .get_layer(config.BACKBONE_LOW_FEATURE_MAP_NAME)
                      .output)
        aspp_input = (base
                      .get_layer(config.BACKBONE_HIGH_FEATURE_MAP_NAME)
                      .output)
    else:
        raise ValueError(f"{config.BACKBONE} is not implemented")

    # ASPP Module Network 구성하기
    encoded_fmap = aspp_module(aspp_input,
                               num_features=config.ASPP_NUM_FEATURES,
                               atrous_rate=config.ASPP_ATROUS_RATE,
                               USE_GROUPNORM=config.USE_GROUPNORM,
                               GROUPS=config.GROUPS)

    # Decoder Network 구성하기
    logits = deeplab_decoder(encoded_fmap, skip_input,
                             num_depth=config.DECODER_NUM_DEPTH,
                             num_features=config.DECODER_NUM_FEATURES,
                             num_skip_features=config.DECODER_NUM_SKIP_FEATURES,
                             use_separable_conv=config.DECODER_USE_SEPARABLE_CONV,
                             USE_GROUPNORM=config.USE_GROUPNORM,
                             GROUPS=config.GROUPS)

    outputs = []
    if config.SEPARATE_BG_CLASSIFIER:
        # BackGround Classifier 분리
        bg_pred = Conv2D(1, (1, 1), activation='sigmoid')(logits)
        bg_pred = ResizeLike(name='bg_prediction')([bg_pred,base.input])
        outputs.append(bg_pred)

    if config.SEPARATE_CRACK_CLASSIFIER:
        # Label Classifier와 Crack Classifier를 분리
        label_pred = Conv2D(config.NUM_CLASSES, (1, 1), activation='softmax')(logits)
        label_pred = ResizeLike(name='label_prediction')([label_pred, base.input])

        abnormal_pred = Conv2D(1, (1, 1), activation='sigmoid')(logits)
        abnormal_pred = ResizeLike(name='crack_prediction')([abnormal_pred, base.input])
        outputs.extend([label_pred, abnormal_pred])
    else:
        # Label Classifier만을 둚
        prediction = Conv2D(config.NUM_CLASSES, (1, 1), name='logits')(logits)
        prediction = ResizeLike(name='label_prediction')([prediction, base.input])
        outputs.append(prediction)

    # Model 선언
    model = Model(inputs, outputs, name='deeplabv3')

    # Freeze the BackBone Network
    if config.BACKBONE_FREEZE:
        freeze_flag = True
        for layer in model.layers:
            if freeze_flag:
                layer.trainable = False
            else:
                layer.trainable = True

            if layer.name == config.BACKBONE_HIGH_FEATURE_MAP_NAME:
                # model train the weights After BACKBONE Network
                freeze_flag = False

    return model


