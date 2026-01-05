"""
TensorFlow/Keras implementation of ResNet-92 baseline.

This is the ResNet trunk corresponding to Attention-92 (no attention modules).
Attention-92 tries to match this architecture exactly, but with attention modules added.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeNormal

try:
    from .layers_tf import PreActBottleneck, make_preact_layer
except ImportError:
    from layers_tf import PreActBottleneck, make_preact_layer


class ResNet92(keras.Model):

    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        block = PreActBottleneck


        he_init = HeNormal()
        self.conv1 = layers.Conv2D(
            64, kernel_size=7, strides=2, padding='same', 
            use_bias=False, kernel_initializer=he_init
        )
        self.bn1 = layers.BatchNormalization()
        self.pool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

        in_ch = 64

        # stage1
        # Pre: 1 
        self.pre1, in_ch = make_preact_layer(block, in_ch, 64, blocks=1)
        # Trunk3 
        self.trunk1, in_ch = make_preact_layer(block, in_ch, 64, blocks=3)
        # Post1 
        self.post1, in_ch = make_preact_layer(block, in_ch, 64, blocks=1)

        # stage 2
        # Pre1 
        self.pre2, in_ch = make_preact_layer(block, in_ch, 128, blocks=1, stride=2)
        # Trunk3 
        self.trunk2, in_ch = make_preact_layer(block, in_ch, 128, blocks=3)
        # Post1 
        self.post2, in_ch = make_preact_layer(block, in_ch, 128, blocks=1)

        # stage 3
        # Pre: 1 
        self.pre3, in_ch = make_preact_layer(block, in_ch, 256, blocks=1, stride=2)
        # Trunk: 3 
        self.trunk3, in_ch = make_preact_layer(block, in_ch, 256, blocks=3)
        # Post: 1 
        self.post3, in_ch = make_preact_layer(block, in_ch, 256, blocks=1)

        # stage 4
        # 3 bottleneck blocks (stride=2 on first block for downsampling)
        self.stage4, in_ch = make_preact_layer(block, in_ch, 512, blocks=3, stride=2)

        # head
        self.bn_head = layers.BatchNormalization()
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, kernel_initializer=HeNormal())

    def call(self, inputs, training=None):
        # Stem
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        # Stage 1
        x = self.pre1(x, training=training)
        x = self.trunk1(x, training=training)
        x = self.post1(x, training=training)

        # Stage 2
        x = self.pre2(x, training=training)
        x = self.trunk2(x, training=training)
        x = self.post2(x, training=training)

        # Stage 3
        x = self.pre3(x, training=training)
        x = self.trunk3(x, training=training)
        x = self.post3(x, training=training)

        # Stage 4
        x = self.stage4(x, training=training)

        # Classifier head
        x = self.bn_head(x, training=training)
        x = tf.nn.relu(x)
        x = self.avgpool(x)
        x = self.fc(x)
        
        x = tf.cast(x, tf.float32)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
        })
        return config


if __name__ == "__main__":
    model = ResNet92()
    x = tf.random.normal((1, 32, 32, 3))
    y = model(x)
    print(y.shape)
