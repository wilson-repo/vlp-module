import keras
from keras.src.applications import imagenet_utils
from VLP import ProjectionLayer


def depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def mbv3_se_block(inputs, filters, se_ratio, block_idx):
    x = keras.layers.GlobalAveragePooling2D(keepdims=True)(inputs)
    x = keras.layers.Conv2D(depth(filters * se_ratio),
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            use_bias=True,
                            name=f"squeeze_excite_expand_pointwise_{block_idx}")(x)
    x = keras.layers.ReLU(name=f"squeeze_excite_expand_act_{block_idx}")(x)
    x = keras.layers.Conv2D(filters,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            use_bias=True,
                            name=f"squeeze_excite_project_pointwise_{block_idx}")(x)
    x = keras.layers.ReLU(6.0, name=f"squeeze_excite_project_act_{block_idx}")(x + 3.0) * (1.0 / 6.0)
    x = keras.layers.Multiply(name=f"mult_{block_idx}")([inputs, x])

    return x


def mbv3_depthwise_separable_conv(x, filters, kernel_size, strides, t, se_ratio, block_idx, activation="hswish"):
    """
    Depthwise Separable Convolution: Depthwise Convolution + Pointwise Convolution (1x1 Conv)
    """
    in_channels = x.shape[-1]

    # Depth-wise convolution
    x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                     depthwise_initializer=keras.initializers.GlorotUniform(),
                                     strides=strides,
                                     padding="same" if strides == (1, 1) else "valid",
                                     use_bias=False,
                                     name=f"expand_depthwise_{block_idx}")(x)
    x = keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name=f"expand_depthwise_bn_{block_idx}")(
        x)
    if activation != "hswish":
        x = keras.layers.ReLU(name=f"expand_depthwise_act_{block_idx}")(x)
    else:
        x = keras.layers.Activation("hard_swish", name=f"expand_depthwise_act_{block_idx}")(x)

    # Squeeze-and-excite module
    if se_ratio:
        x = mbv3_se_block(x, in_channels, se_ratio, block_idx)

    # Pointwise convolution (1x1 convolution)
    x = keras.layers.Conv2D(filters, kernel_size=(1, 1),
                            kernel_initializer=keras.initializers.GlorotUniform(),
                            strides=(1, 1),
                            padding="same",
                            use_bias=False,
                            name=f"project_pointwise_{block_idx}")(x)
    x = keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999,
                                        name=f"project_pointwise_bn_{block_idx}")(x)

    return x


def mbv3_bottleneck(x, filters, kernel_size, strides, t, se_ratio, block_idx, exp_var, activation="hswish"):
    """
    MobileNetV3 Bottleneck Block
    """
    in_channels = x.shape[-1]
    x_shortcut = x
    # Expanding phase (1x1 convolution)
    if exp_var is not None:
        x = ProjectionLayer(projection_var=exp_var, name=f"VLP_{block_idx}")(x)
    else:
        x = keras.layers.Conv2D(depth(in_channels * t), kernel_size=(1, 1),
                                kernel_initializer=keras.initializers.GlorotUniform(),
                                strides=(1, 1),
                                padding='same',
                                use_bias=False,
                                name=f"expand_pointwise_{block_idx}")(x)
    x = keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name=f"expand_pointwise_bn_{block_idx}")(
        x)
    if activation != "hswish":
        x = keras.layers.ReLU(name=f"expand_pointwise_act_{block_idx}")(x)
    else:
        x = keras.layers.Activation("hard_swish", name=f"expand_pointwise_act_{block_idx}")(x)

    if strides == (2, 2):
        x = keras.layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, kernel_size),
                                       name=f"expand_depthwise_pad_{block_idx}")(x)

    # Depthwise separable convolution
    x = mbv3_depthwise_separable_conv(x, filters, kernel_size, strides, t, se_ratio, block_idx, activation=activation)

    # Skip connection if input and output dimensions match
    if in_channels == filters and strides == (1, 1):
        x = keras.layers.Add(name=f"expand_add_{block_idx}")([x_shortcut, x])

    return x


def mod_MobileNetV3(input_shape, num_classes, exp_var):
    """
    Custom MobileNetV3 Implementation
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)

    # First Conv Layer (Standard Convolution)
    x = keras.layers.Conv2D(16, kernel_size=(3, 3),
                            kernel_initializer=keras.initializers.GlorotUniform(),
                            strides=(2, 2),
                            padding='same',
                            use_bias=False,
                            name='main_conv')(x)
    x = keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='main_conv_bn')(x)
    x = keras.layers.Activation("hard_swish", name='main_conv_act')(x)
    x_shortcut = x
    x = keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                     depthwise_initializer=keras.initializers.GlorotUniform(),
                                     strides=(1, 1),
                                     padding='same',
                                     use_bias=False,
                                     name='main_depthwise')(x)
    x = keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='main_depthwise_bn')(x)
    x = keras.layers.ReLU(name='main_depthwise_act')(x)
    x = keras.layers.Conv2D(16, kernel_size=(1, 1),
                            kernel_initializer=keras.initializers.GlorotUniform(),
                            strides=(1, 1),
                            padding='same',
                            use_bias=False,
                            name='main_project')(x)
    x = keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='main_project_bn')(x)
    x = keras.layers.Add(name='main_add')([x_shortcut, x])

    x = mbv3_bottleneck(x, filters=24, kernel_size=(3, 3), strides=(2, 2), t=4, se_ratio=None, block_idx=1,
                        exp_var=exp_var, activation="relu")  # 1st bottleneck
    x = mbv3_bottleneck(x, filters=24, kernel_size=(3, 3), strides=(1, 1), t=3, se_ratio=None, block_idx=2,
                        exp_var=exp_var,activation="relu")  # 2nd bottleneck

    x = mbv3_bottleneck(x, filters=40, kernel_size=(5, 5), strides=(2, 2), t=3, se_ratio=0.25, block_idx=3,
                        exp_var=exp_var,activation="relu")  # 3rd bottleneck
    x = mbv3_bottleneck(x, filters=40, kernel_size=(5, 5), strides=(1, 1), t=3, se_ratio=0.25, block_idx=4,
                        exp_var=exp_var,activation="relu")  # 4th bottleneck
    x = mbv3_bottleneck(x, filters=40, kernel_size=(5, 5), strides=(1, 1), t=3, se_ratio=0.25, block_idx=5,
                        exp_var=exp_var,activation="relu")  # 5th bottleneck

    x = mbv3_bottleneck(x, filters=80, kernel_size=(3, 3), strides=(2, 2), t=6, se_ratio=None, exp_var=exp_var,
                        block_idx=6)  # 6th bottleneck
    x = mbv3_bottleneck(x, filters=80, kernel_size=(3, 3), strides=(1, 1), t=2.5, se_ratio=None, exp_var=exp_var,
                        block_idx=7)  # 7th bottleneck
    x = mbv3_bottleneck(x, filters=80, kernel_size=(3, 3), strides=(1, 1), t=2.3, se_ratio=None, exp_var=exp_var,
                        block_idx=8)  # 8th bottleneck
    x = mbv3_bottleneck(x, filters=80, kernel_size=(3, 3), strides=(1, 1), t=2.3, se_ratio=None, exp_var=exp_var,
                        block_idx=9)  # 9th bottleneck

    x = mbv3_bottleneck(x, filters=112, kernel_size=(3, 3), strides=(1, 1), t=6, se_ratio=0.25, exp_var=exp_var,
                        block_idx=10)  # 10th bottleneck
    x = mbv3_bottleneck(x, filters=112, kernel_size=(3, 3), strides=(1, 1), t=6, se_ratio=0.25, exp_var=exp_var,
                        block_idx=11)  # 11th bottleneck
    # x = bottleneck(x, filters=96, kernel_size=(3, 3), strides=(1, 1), t=6)  # 12th bottleneck

    x = mbv3_bottleneck(x, filters=160, kernel_size=(5, 5), strides=(2, 2), t=6, se_ratio=0.25, exp_var=exp_var,
                        block_idx=12)  # 13th bottleneck
    x = mbv3_bottleneck(x, filters=160, kernel_size=(5, 5), strides=(1, 1), t=6, se_ratio=0.25, exp_var=exp_var,
                        block_idx=13)  # 14th bottleneck
    x = mbv3_bottleneck(x, filters=160, kernel_size=(5, 5), strides=(1, 1), t=6, se_ratio=0.25, exp_var=exp_var,
                        block_idx=14)  # 15th bottleneck

    # x = bottleneck(x, filters=320, kernel_size=(3, 3), strides=(1, 1), t=6)  # Final bottleneck

    # Final layers (global average pooling and dense layer)
    last_conv_ch = depth(x.shape[-1] * 6)
    # x = keras.layers.Dropout(rate=0.3)(x)
    x = keras.layers.Conv2D(last_conv_ch, kernel_size=(1, 1),
                            kernel_initializer=keras.initializers.GlorotUniform(),
                            strides=(1, 1),
                            padding='same',
                            use_bias=False,
                            name="last_conv")(x)
    x = keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name="last_conv_bn")(x)
    x = keras.layers.Activation("hard_swish", name="last_conv_act")(x)

    x = keras.layers.GlobalAveragePooling2D(keepdims=True)(x)

    x = keras.layers.Conv2D(1280, kernel_size=(1, 1),
                            kernel_initializer=keras.initializers.GlorotUniform(),
                            strides=(1, 1),
                            padding="same",
                            use_bias=True,
                            name="last_conv_2")(x)
    x = keras.layers.Activation("hard_swish", name="last_conv_act_2")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(num_classes, kernel_size=(1, 1),
                            padding="same",
                            use_bias=True,
                            name="logits")(x)
    x = keras.layers.Flatten()(x)

    output = keras.layers.Activation(activation="softmax", name="predictions")(x)

    return keras.models.Model(inputs=inputs, outputs=output, name="mod_MobileNetV3")


if __name__ == '__main__':
    # Defining model and configurations
    # num_classes: The number of classification classes
    # exp_var: The lambda parameter used to define VLP modules
    mod_model = mod_MobileNetV3(input_shape=(224, 224, 3), num_classes=10, exp_var=1.0)