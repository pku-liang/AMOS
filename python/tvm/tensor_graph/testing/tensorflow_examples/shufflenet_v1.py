import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import layers
import numpy as np

import time
import argparse

def channel_shuffle(inputs, num_groups):
    
    n, h, w, c = inputs.shape
    x_reshaped = tf.reshape(inputs, [-1, h, w, num_groups, c // num_groups])
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(x_transposed, [-1, h, w, c])
    
    return output


def group_conv(inputs, filters, kernel, strides, num_groups):
    
    conv_side_layers_tmp = tf.split(inputs, num_groups ,axis=3)
    conv_side_layers = []
    for layer in conv_side_layers_tmp:
        conv_side_layers.append(tf.keras.layers.Conv2D(filters//num_groups, kernel, strides, padding='same')(layer))
    x = tf.concat(conv_side_layers, axis=-1)
    
    return x

def conv(inputs, filters, kernel_size, stride=1, activation=False):
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation:
        x = tf.keras.layers.Activation('relu')(x)
        
    return x

def depthwise_conv_bn(inputs, kernel_size, stride=1):
    
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, 
                                        strides=stride, 
                                        padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
        
    return x

class ShuffleNetUnitA(layers.Layer):
    def __init__(self, num_groups):
        super(ShuffleNetUnitA, self).__init__()
        self.num_groups = num_groups
    
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.out_channels = self.in_channels
        self.bottleneck_channels = self.out_channels // 4
    
    def call(self, inputs):
        x = group_conv(inputs, self.bottleneck_channels, kernel=1, strides=1, num_groups=self.num_groups)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = channel_shuffle(x, self.num_groups)
        x = depthwise_conv_bn(x, kernel_size=3, stride=1)
        x = tf.keras.layers.BatchNormalization()(x)
        x = group_conv(x, self.out_channels, kernel=1, strides=1, num_groups=self.num_groups)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.add([inputs, x])
        x = tf.keras.layers.Activation('relu')(x)

        return x

class ShuffleNetUnitB(layers.Layer):
    def __init__(self, out_channels, num_groups):
        super(ShuffleNetUnitB, self).__init__()
        self.out_channels = out_channels
        self.num_groups = num_groups
    
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.out_channels -= self.in_channels
        self.bottleneck_channels = self.out_channels // 4
    
    def call(self, inputs):
        x = group_conv(inputs, self.bottleneck_channels, kernel=1, strides=1, num_groups=self.num_groups)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = channel_shuffle(x, self.num_groups)
        x = depthwise_conv_bn(x, kernel_size=3, stride=2)
        x = group_conv(x, self.out_channels, kernel=1, strides=1, num_groups=self.num_groups)
        x = tf.keras.layers.BatchNormalization()(x)
        y = tf.keras.layers.AvgPool2D(pool_size=3, strides=2, padding='same')(inputs)
        x = tf.concat([y, x], axis=-1)
        x = tf.keras.layers.Activation('relu')(x)

        return x

def stage(inputs, out_channels, num_groups, n):
    
    x = ShuffleNetUnitB(out_channels, num_groups)(inputs)
    
    for _ in range(n):
        x = ShuffleNetUnitA(num_groups)(x)
    return x

def ShuffleNet(inputs, first_stage_channels=240, num_groups=3):
    x = tf.keras.layers.Conv2D(filters=24, 
                               kernel_size=3, 
                               strides=2, 
                               padding='same')(inputs)
    x = tf.keras.layers.AvgPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    x = stage(x, first_stage_channels, num_groups, n=3)
    x = stage(x, first_stage_channels*2, num_groups, n=7)
    x = stage(x, first_stage_channels*4, num_groups, n=3)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1000)(x)
    
    return x, None

if __name__ == "__main__":
    model = ShuffleNet
    data_val = np.random.randn(1, 224, 224, 3)

    data = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    out, _ = model(data)

    values = {
        data: data_val
    }

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    out_val = sess.run([out], feed_dict=values)

    print(out_val)