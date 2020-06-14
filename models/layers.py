"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Concatenate, UpSampling2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, ZeroPadding2D, DepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects
from .normalization import GroupNormalization


class MeanShift(Layer):
    """MobileNet의 Input 형태로 바꾸어주는 Layer
    1. (0,255) -> (-1,1)
    2. BGR -> RGB
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        x = x / 127.5 - 1.
        return K.reverse(x, axes=-1)

    def compute_output_shape(self, input_shape):
        return input_shape


class ResizeLike(Layer):
    """Change the size of tensor(height & width) to target node"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        assert isinstance(inputs, list)
        input_node = inputs[0]
        target_node = inputs[1]
        if target_node.shape[1] is None:
            target_shape = tf.shape(target_node)
        else:
            target_shape = target_node.shape
        target_size = (target_shape[1], target_shape[2])

        resize = tf.compat.v1.image.resize
        return resize(input_node, target_size,
                      align_corners=True)

    def compute_output_shape(self, input_shapes):
        assert isinstance(input_shapes, list)
        input_shape, target_shape = input_shapes
        return (input_shape[0], target_shape[1], target_shape[2], input_shape[-1])


class GlobalAveragePoolingLayer(Layer):
    """
    Global Average Pooling Layer을 tf.reduce_mean으로 구현

    CAUTION
    tf.reduce_mean은 keras에서 json serialization이 적용되지 못함

    reference : https://stackoverflow.com/questions/55510586/cant-save-keras-model-when-using-reduce-mean-in-the-model
    그렇기 때문에, 우회해서 해결해야 하는데, 그 방법으로 Lambda로 씌어주거나, 위와 같이 GlobalAveragePooling class을 만들어서 적용해주어야 함


    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, **kwargs):
        return tf.reduce_mean(x, axis=(1, 2), keepdims=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1, input_shape[-1])


class AtrousSeparableConv2D(Layer):
    """ Atrous Separable Convolution Layer

        Operation Order:
            Atrous & Depthwise Convolution ->
            Batch Normalization ->
            Activation ->
            Pointwise Convolution ->
            Activation
    """

    def __init__(self, filters, stride=1, kernel_size=3, dilation_rate=3,
                 use_groupnorm=False, groups=16, **kwargs):
        prefix = kwargs.get('name', 'AtrousSeparableConv2d')
        super().__init__(**kwargs)
        self.use_groupnorm = use_groupnorm
        self.groups = groups
        self.filter = filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            self.padding_layer = ZeroPadding2D((pad_beg, pad_end))
            depth_padding = 'valid'

        self.depth_conv2d = DepthwiseConv2D((kernel_size, kernel_size), (stride, stride),
                                            dilation_rate=dilation_rate, padding=depth_padding,
                                            use_bias=False, name=prefix + '_depthwise')
        self.point_conv2d = Conv2D(filters, (1, 1), use_bias=False,
                                   name=prefix + '_pointwise')
        if self.use_groupnorm:
            self.depth_norm = GroupNormalization(groups=self.groups, name=prefix + '_depthwise_GN')
            self.point_norm = GroupNormalization(groups=self.groups, name=prefix + '_pointwise_GN')
        else:
            self.depth_norm = BatchNormalization(epsilon=1e-5, name=prefix + '_depthwise_BN')
            self.point_norm = BatchNormalization(epsilon=1e-5, name=prefix + '_pointwise_BN')

        self.depth_relu = ReLU(name=prefix+'_depthwise_relu')
        self.point_relu = ReLU(name=prefix+'_pointwise_relu')

    def call(self, inputs, **kwargs):
        if self.stride != 1:
            inputs = self.padding_layer(inputs)

        x = self.depth_conv2d(inputs)
        x = self.depth_norm(x)
        x = self.depth_relu(x)

        x = self.point_conv2d(x)
        x = self.point_norm(x)
        x = self.point_relu(x)
        return x

    def get_config(self):
        config = {'filters': self.filter,
                  'stride': self.stride,
                  'kernel_size': self.kernel_size,
                  'dilation_rate': self.dilation_rate,
                  'use_groupnorm': self.use_groupnorm,
                  'groups': self.groups}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def aspp_module(inputs, num_features=256, atrous_rate=(6, 12, 18), **kwargs):
    use_groupnorm = kwargs.get('USE_GROUPNORM', False)
    groups = kwargs.get('GROUPS', 16)

    aspp_branches = []
    # ASPP 1x1 Branch
    aspp = Conv2D(num_features, (1, 1), use_bias=False, name='aspp_1x1')(inputs)
    if use_groupnorm:
        aspp = GroupNormalization(groups=groups, name='aspp_1x1_GN')(aspp)
    else:
        aspp = BatchNormalization(name='aspp_1x1_BN', epsilon=1e-5)(aspp)
    aspp = ReLU(name='aspp_1x1_relu')(aspp)
    aspp_branches.append(aspp)

    # ASPP Branches
    for rate in atrous_rate:
        aspp = AtrousSeparableConv2D(num_features, dilation_rate=rate,
                                     use_groupnorm=use_groupnorm,
                                     groups=groups, name=f'aspp_{rate}')(inputs)
        aspp_branches.append(aspp)

    # ASPP Pooling Branch
    aspp = GlobalAveragePoolingLayer()(inputs)
    aspp = Conv2D(num_features, (1, 1), use_bias=False,
                  name='aspp_pool_conv')(aspp)
    aspp = ReLU(name='aspp_pool_relu')(aspp)
    aspp = ResizeLike()([aspp,inputs])

    aspp_branches.append(aspp)

    # Concatenation & projection
    aspp_concat = Concatenate(name='aspp_concat')(aspp_branches)
    x = Conv2D(num_features, (1,1), use_bias=False,
               name='concat_projection')(aspp_concat)
    if use_groupnorm:
        x = GroupNormalization(groups=groups, name='concat_projection_GN')(x)
    else:
        x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = ReLU(name='concat_projection_relu')(x)
    return x


def deeplab_decoder(inputs,
                    skip_input=None,
                    num_depth=2,
                    num_features=256,
                    num_skip_features=48,
                    use_separable_conv=False,
                    **kwargs):
    use_groupnorm = kwargs.get('USE_GROUPNORM', False)
    groups = kwargs.get('GROUPS', 16)

    if skip_input is None:
        x = UpSampling2D((4, 4), interpolation='bilinear')(inputs)
    else:
        skip = Conv2D(num_skip_features, (1, 1), use_bias=False,
                      name='skip_projection')(skip_input)
        if use_groupnorm:
            skip = GroupNormalization(groups=groups, name='skip_projection_GN')(skip)
        else:
            skip = BatchNormalization(name='skip_projection_BN')(skip)

        skip = ReLU(name='skip_projection_relu')(skip)

        upsampled = ResizeLike()([inputs,skip])
        x = Concatenate()([upsampled, skip])

    for i in range(num_depth):
        if use_separable_conv:
            x = AtrousSeparableConv2D(num_features,
                                      use_groupnorm=use_groupnorm,
                                      groups=groups,
                                      name=f'decoder_conv{i}')(x)
        else:
            x = Conv2D(num_features, (3, 3), use_bias=False,
                       padding='same', name=f'decoder_conv{i}')(x)
            if use_groupnorm:
                x = GroupNormalization(groups=groups,
                                       name=f'decoder_conv{i}_GN')(x)
            else:
                x = BatchNormalization(name=f'decoder_conv{i}_BN',
                                       epsilon=1e-5)(x)
            x = ReLU(name=f'decoder_conv{i}_relu')(x)
    return x


# register custom layer class
get_custom_objects().update({
    'MeanShift' : MeanShift,
    'ResizeLike' : ResizeLike,
    'GlobalAveragePoolingLayer' : GlobalAveragePoolingLayer,
    'AtrousSeparableConv2D': AtrousSeparableConv2D
}
)