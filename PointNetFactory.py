import numpy as np
from OrthogonalRegularizer import OrthogonalRegularizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PointNetFactory:
    def __init__(self):
        self.model = []

    def conv_bn(self, x, filters):
        x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def dense_bn(self, x, filters):
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def tnet(self, inputs, num_features):
        # Initalise bias as the indentity matrix
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)

        x = self.conv_bn(inputs, 64)
        x = self.conv_bn(x, 128)
        x = self.conv_bn(x, 1024)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 512)
        x = self.dense_bn(x, 256)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        # Apply affine transformation to input features
        return layers.Dot(axes=(2, 1))([inputs, feat_T])

    def create_model(self, num_points, num_point_axis, num_classes):
        inputs = keras.Input(shape=(num_points, num_point_axis))
        x = self.tnet(inputs, num_point_axis)
        x = self.conv_bn(x, 64)
        x = self.conv_bn(x, 64)
        x = self.tnet(x, 64)
        x = self.conv_bn(x, 64)
        x = self.conv_bn(x, 128)
        x = self.conv_bn(x, 1024)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 512)
        x = layers.Dropout(0.3)(x)
        x = self.dense_bn(x, 256)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
        self.model = model
        return model
