"""
TensorFlow/Keras implementation of Residual Attention Network layers.

This module contains the core building blocks:
- PreActBottleneck: Pre-activation bottleneck block
- MaskBranch: U-Net style attention mask branch
- AttentionModule: Trunk + Mask + Fusion module (followed by the description of the standatard paper we are given)
- reference paper: https://arxiv.org/pdf/1704.06904
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeNormal


# Pre-activation Bottleneck Block: N -> ReLU -> Conv
class PreActBottleneck(layers.Layer):
    expansion = 4  

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, **kwargs):
        super().__init__(**kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

        mid = out_ch  

        # Pre-activation BN + ReLU before each conv
        he_init = HeNormal()
        
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(
            mid, kernel_size=1, strides=1, padding='same', 
            use_bias=False, kernel_initializer=he_init
        )

        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            mid, kernel_size=3, strides=stride, padding='same', 
            use_bias=False, kernel_initializer=he_init
        )

        self.bn3 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(
            out_ch * self.expansion, kernel_size=1, strides=1,
            padding='same', use_bias=False, kernel_initializer=he_init
        )

        self.downsample = downsample

    # forward pass
    def call(self, inputs, training=None):
        out = self.bn1(inputs, training=training)
        out = tf.nn.relu(out)


        shortcut = inputs
        if self.downsample is not None:
            shortcut = self.downsample(out)

        # 1x1 -> 3x3 -> 1x1
        out = self.conv1(out)
        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn3(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv3(out)

        return out + shortcut

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'in_ch': self.in_ch,
            'out_ch': self.out_ch,
            'stride': self.stride,
        })
        return config


# ResNet stage
class PreActLayer(layers.Layer):
    def __init__(self, block_class, in_ch, out_ch, blocks, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.block_list = []
        
        downsample = None
        
        he_init = HeNormal()
        if stride != 1 or in_ch != out_ch * block_class.expansion:  # Projection shortcut if channel count changes or stride != 1
            downsample = layers.Conv2D(
                out_ch * block_class.expansion,
                kernel_size=1,
                strides=stride,
                padding='same',
                use_bias=False,
                kernel_initializer=he_init
            )
        
        # First block performs downsampling if needed
        first_block = block_class(in_ch, out_ch, stride=stride, downsample=downsample)
        self.block_list.append(first_block)
        in_ch = out_ch * block_class.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            self.block_list.append(block_class(in_ch, out_ch))
    
    def call(self, inputs, training=None):
        x = inputs
        for block in self.block_list:
            x = block(x, training=training)
        return x


def make_preact_layer(block_class, in_ch, out_ch, blocks, stride=1):
 
    layer = PreActLayer(block_class, in_ch, out_ch, blocks, stride)
    final_ch = out_ch * block_class.expansion
    return layer, final_ch


# Mask Branch (bottom-up + top-down attention mask)
class MaskBranch(layers.Layer):
    """
    - bottom-up: residual blocks + pooling
    - bottoleneck: deeper residual blocks
    - top-down: upsampling + skip connections + residual blocks
    - Final 1x1 conv + sigmoid to produce mask
    """
    def __init__(self, block_class, ch, depth=2, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.ch = ch
        self.block_class = block_class

        # bottom-up
        self.down_blocks = []
        self.pools = []

        for i in range(depth):
            # Each down block contains 2 bottleneck blocks
            down_block_list = [
                block_class(ch, ch // block_class.expansion),
                block_class(ch, ch // block_class.expansion)
            ]
            self.down_blocks.append(down_block_list)
            self.pools.append(layers.MaxPooling2D(pool_size=2, strides=2))

        # bottleneck
        self.mid = [
            block_class(ch, ch // block_class.expansion),
            block_class(ch, ch // block_class.expansion)
        ]

        # up-down
        self.up_blocks = []
        for i in range(depth):
            up_block_list = [
                block_class(ch, ch // block_class.expansion),
                block_class(ch, ch // block_class.expansion)
            ]
            self.up_blocks.append(up_block_list)

        # conv
        he_init = HeNormal()
        self.conv_mask = layers.Conv2D(
            ch, kernel_size=1, padding='same', use_bias=False,
            kernel_initializer=he_init
        )

    # forward pass for mask branch
    def call(self, inputs, training=None):
        skips = []
        out = inputs

        # bottom-up
        for down_block_list, pool in zip(self.down_blocks, self.pools):
            for block in down_block_list:
                out = block(out, training=training)
            skips.append(out) 
            out = pool(out)

        # bottleneck
        for block in self.mid:
            out = block(out, training=training)

        # top-down
        for up_block_list in self.up_blocks:
            # Upsample using bilinear interpolation
            current_shape = tf.shape(out)
            h = current_shape[1] * 2
            w = current_shape[2] * 2
            out = tf.image.resize(out, [h, w], method='bilinear')

            # Skip connection
            skip = skips.pop()  
            
            out = out + tf.cast(skip, out.dtype)

            
            for block in up_block_list:
                out = block(out, training=training)

       
        out = self.conv_mask(out)
        return tf.nn.sigmoid(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            'ch': self.ch,
            'depth': self.depth,
        })
        return config


# attention module (trunk + mask + fusion)
class AttentionModule(layers.Layer):
    def __init__(self, block_class, ch, trunk_blocks=2, mask_depth=2, att_type="arl", **kwargs):
        # att_type: arl for attention residual, nal for naive attention learning
        super().__init__(**kwargs)
        self.ch = ch
        self.trunk_blocks = trunk_blocks
        self.mask_depth = mask_depth
        self.block_class = block_class
        self.att_type = att_type
        
        if att_type not in ["arl", "nal"]:
            raise ValueError(f"Invalid att_type: {att_type}. Must be 'arl' or 'nal'")

        # Trunk: sequence of bottleneck blocks
        self.trunk = []
        for _ in range(trunk_blocks):
            self.trunk.append(block_class(ch, ch // block_class.expansion))

        # Mask branch
        self.mask = MaskBranch(block_class, ch, depth=mask_depth)

        # Post-fusion bottleneck
        self.post = block_class(ch, ch // block_class.expansion)


    def call(self, inputs, training=None):
        # Trunk branch
        t = inputs
        for block in self.trunk:
            t = block(t, training=training)

        # Mask branch
        m = self.mask(inputs, training=training)

  
        if self.att_type == "arl":         # H(x) = (1 + M(x)) * T(x)
            out = (1.0 + m) * t
        else:                              # H(x) = M(x) * T(x)
            out = m * t

        
        out = self.post(out, training=training)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'ch': self.ch,
            'trunk_blocks': self.trunk_blocks,
            'mask_depth': self.mask_depth,
            'att_type': self.att_type,
        })
        return config

