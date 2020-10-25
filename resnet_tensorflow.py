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
        self.activation = keras.layers.Activation("relu")
        self.layers = [
            keras.layers.Conv2D(filters, kernel_size=1,
                                strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, kernel_size=3,
                                strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters * 4, kernel_size=1,
                                strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
        ]

        if strides > 1:
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


class ResNet34(keras.models.Model):
    def __init__(self):
        super(ResNet34, self).__init__()

        self.base = [
            keras.layers.Conv2D(64, 7, strides=2, input_shape=[
                                224, 224, 3], padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPool2D(
                pool_size=3, strides=2, padding="same")
        ]

        self.residual_blocks = []
        prev_filters = 64
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            self.residual_blocks.append(
                ResidualBlock(filters, strides=strides))
            prev_filters = filters

        self.global_avg_pool = keras.layers.GlobalAvgPool2D()
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = inputs
        for layer in self.base:
            x = layer(x)

        for layer in self.residual_blocks:
            x = layer(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x


def test():
    x = tf.random.normal([2, 3, 224, 224])
    model = ResNet34()
    output = model(x)
    print(output.shape)
    print(model.summary())


test()
