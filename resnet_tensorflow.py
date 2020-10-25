import tensorflow as tf
from tensorflow import keras


class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.layers.Activation("relu")
        self.layers = [
            keras.layers.Conv2D(filters, kernel_size=3, strides=strides,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, kernel_size=3, strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, kernel_size=1,
                                    strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        z = inputs
        for layer in self.layers:
            z = layer(z)

        skip_z = inputs
        for layer in self.skip_layers:
            skip_z = layer(skip_z)

        return self.activation(z + skip_z)


class BottleneckBlock(keras.layers.Layer):
    def __init__(self, filters, strides, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.layers.Activation("relu")
        self.layers = [
            keras.layers.Conv2D(filters, kernel_size=1,
                                strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, kernel_size=3,
                                strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters * 4, kernel_size=1,
                                strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1 or filters != filters * 4:
            self.skip_layers = [
                keras.layers.Conv2D(
                    filters * 4, kernel_size=1, strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        z = inputs
        for layer in self.layers:
            z = layer(z)

        skip_z = inputs
        for layer in self.skip_layers:
            skip_z = layer(skip_z)

        return self.activation(z + skip_z)


class ResNet(keras.models.Model):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.base = [
            keras.layers.Conv2D(64, 7, strides=2, input_shape=[
                                224, 224, 3], padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPool2D(
                pool_size=3, strides=2, padding="same")
        ]

        # Residual layers
        self.conv_2 = self._make_layer(block, layers[0], 64, strides=1)
        self.conv_3 = self._make_layer(block, layers[1], 128, strides=2)
        self.conv_4 = self._make_layer(block, layers[2], 256, strides=2)
        self.conv_5 = self._make_layer(block, layers[3], 512, strides=2)

        self.global_avg_pool = keras.layers.GlobalAvgPool2D()
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(10, activation="softmax")

    def _make_layer(self, block, num_blocks, filters, strides):
        layers = []

        layers.append(block(filters, strides=strides))
        for _ in range(num_blocks - 1):
            layers.append(block(filters, strides=1))

        return layers

    def call(self, inputs):
        x = inputs

        for layer in self.base:
            x = layer(x)

        for layer in self.conv_2:
            x = layer(x)

        for layer in self.conv_3:
            x = layer(x)

        for layer in self.conv_4:
            x = layer(x)

        for layer in self.conv_5:
            x = layer(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x


def make_resnet_34():
    model = ResNet(ResidualBlock, [3, 4, 6, 3])

    return model


def make_resnet_50():
    model = ResNet(BottleneckBlock, [3, 4, 6, 3])

    return model


def make_resnet_101():
    model = ResNet(BottleneckBlock, [3, 4, 23, 3])

    return model


def make_resnet_152():
    model = ResNet(BottleneckBlock, [3, 8, 36, 3])

    return model


def test(resnet_version):
    x = tf.random.normal([2, 3, 224, 224])

    if resnet_version == 'resnet34':
        model = make_resnet_34()
    if resnet_version == 'resnet50':
        model = make_resnet_50()
    elif resnet_version == 'resnet101':
        model = make_resnet_101()
    elif resnet_version == 'resnet152':
        model = make_resnet_152()
    else:
        print("Please select a valid version")

    output = model(x)

    print(output.shape)
    print(model.summary())


test('resnet152')
