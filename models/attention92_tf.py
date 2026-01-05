import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeNormal
from .layers_tf import PreActBottleneck, MaskBranch, AttentionModule, make_preact_layer



# Residual Attention Network — Attention-92 (ImageNet version)
class ResidualAttentionModel92(keras.Model):
    """
    Architecture:
    - Stem: 7x7 conv + maxpool
    - Stage 1: pre -> attention -> post
    - Stage 2: pre -> attention -> post
    - Stage 3: pre -> attention -> post
    - Stage 4: standard ResNet bottlenecks (no attention)
    - Classification head

    """
    def __init__(self, num_classes=1000, att_type="arl", **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.att_type = att_type
        block = PreActBottleneck

        # STEM
        # He/Kaiming initialization for all Conv2D layers (as per paper)
        he_init = HeNormal()
        self.conv1 = layers.Conv2D(
            64, kernel_size=7, strides=2, padding='same', 
            use_bias=False, kernel_initializer=he_init
        )
        self.bn1 = layers.BatchNormalization()
        self.pool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

        in_ch = 64

        # STAGE 1
        # Pre: 1 bottleneck block
        self.pre1, in_ch = make_preact_layer(block, in_ch, 64, blocks=1)
        
        # 2 attention modules
        self.att1 = keras.Sequential([
            AttentionModule(block, in_ch, trunk_blocks=3, mask_depth=2, att_type=att_type),
            AttentionModule(block, in_ch, trunk_blocks=3, mask_depth=2, att_type=att_type),
        ], name="att1_x2")
        # Post: 1 bottleneck block
        self.post1, in_ch = make_preact_layer(block, in_ch, 64, blocks=1)

        # STAGE 2
        # Pre: 1 bottleneck block (stride=2 for downsampling)
        self.pre2, in_ch = make_preact_layer(block, in_ch, 128, blocks=1, stride=2)
        
        # 2 attention modules
        self.att2 = keras.Sequential([
            AttentionModule(block, in_ch, trunk_blocks=3, mask_depth=1, att_type=att_type),
            AttentionModule(block, in_ch, trunk_blocks=3, mask_depth=1, att_type=att_type),
        ], name="att1_x2")
        
        # Post: 1 bottleneck block
        self.post2, in_ch = make_preact_layer(block, in_ch, 128, blocks=1)

        # STAGE 3
        # Pre: 1 bottleneck block (stride=2 for downsampling)
        self.pre3, in_ch = make_preact_layer(block, in_ch, 256, blocks=1, stride=2)
        
       # 2 attention modules
        self.att3 = keras.Sequential([
            AttentionModule(block, in_ch, trunk_blocks=3, mask_depth=1, att_type=att_type),
            AttentionModule(block, in_ch, trunk_blocks=3, mask_depth=1, att_type=att_type),
        ], name="att1_x2")
        
        # Post: 1 bottleneck block
        self.post3, in_ch = make_preact_layer(block, in_ch, 256, blocks=1)

        # STAGE 4 without attention
        # 3 bottleneck blocks (stride=2 on first block for downsampling)
        self.stage4, in_ch = make_preact_layer(block, in_ch, 512, blocks=3, stride=2)

        # head
        self.bn_head = layers.BatchNormalization()
        self.avgpool = layers.GlobalAveragePooling2D()
        # He/Kaiming initialization for Dense layer (as per paper)
        self.fc = layers.Dense(num_classes, kernel_initializer=HeNormal())

    def call(self, inputs, training=None):
        # Stem
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        # Stage 1
        x = self.pre1(x, training=training)
        x = self.att1(x, training=training)
        x = self.post1(x, training=training)

        # Stage 2
        x = self.pre2(x, training=training)
        x = self.att2(x, training=training)
        x = self.post2(x, training=training)

        # Stage 3
        x = self.pre3(x, training=training)
        x = self.att3(x, training=training)
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
            'att_type': self.att_type,
        })
        return config



if __name__ == "__main__":
    model = ResidualAttentionModel92(num_classes=1000)

    # Test with dummy input (ImageNet size)
    x = tf.random.normal((2, 224, 224, 3))
    out = model(x, training=False)

    print("Model created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Count parameters
    total_params = sum(
        tf.size(w).numpy() for w in model.trainable_variables
    )
    print(f"Total trainable parameters: {total_params:,}")

    # Test with CIFAR size (32x32)
    print("\nTesting with CIFAR input size (32x32):")
    x_cifar = tf.random.uniform([1, 32, 32, 3])
    y_cifar = model(x_cifar, training=False)
    print(f"CIFAR input shape: {x_cifar.shape}")
    print(f"CIFAR output shape: {y_cifar.shape}")
    print("Model builds and runs successfully!")

