

import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from models.attention56_tf import ResidualAttentionModel56
from models.attention92_tf import ResidualAttentionModel92
from models.attention128_tf import ResidualAttentionModel128
from models.attention164_tf import ResidualAttentionModel164

from models.attention92_simple import ResidualAttentionModel92_simple
from models.attention56_simple import ResidualAttentionModel56_simple
from models.attention128_simple import ResidualAttentionModel128_simple
from models.attention164_simple import ResidualAttentionModel164_simple



import tensorflow as tf
from tensorflow.keras import metrics

class Top5Accuracy(metrics.Metric):
    def __init__(self,name="top_5_accuracy",**kwargs):
        super().__init__(name=name,dtype=tf.float32,**kwargs)
        self.successful_predictions=self.add_weight(name="successful_predictions",initializer="zeros",dtype=tf.float32)
        self.processed_samples=self.add_weight(name="processed_samples",initializer="zeros",dtype=tf.float32)

    def update_state(self,y_true,y_pred,sample_weight=None):
        y_true_indices=tf.argmax(y_true,axis=1)
        k=5
        _,top_k_indices=tf.nn.top_k(y_pred,k=k)
        y_true_reshaped=tf.reshape(y_true_indices,(-1,1))
        is_correct_hit=tf.reduce_any(top_k_indices==y_true_reshaped,axis=1)
        num_hits=tf.reduce_sum(tf.cast(is_correct_hit,tf.float32))
        num_samples=tf.cast(tf.shape(y_true)[0],tf.float32)
        self.successful_predictions.assign_add(num_hits)
        self.processed_samples.assign_add(num_samples)

    def result(self):
        epsilon=tf.keras.backend.epsilon()
        return self.successful_predictions/(self.processed_samples+epsilon)

    def reset_states(self):
        self.successful_predictions.assign(0.0)
        self.processed_samples.assign(0.0)
# Configuration 

def parse_args():
    """Simple argument parser."""
    parser = argparse.ArgumentParser(description='Train CNN on CIFAR-10')
    parser.add_argument('--model', type=str, default='attention92',
                        choices=['attention56', 'attention92', 'attention128', 'attention164', 'attention92_simple','attention56_simple','attention128_simple','attention164_simple'],
                        help='Model to train: attention56 or attention92 (default: attention56)')
    parser.add_argument('--att_type', type=str,default='arl',
                        choices=['arl','nal'],
                        help="Attention type: 'arl' (default) or 'nal'")
    parser.add_argument('--epochs',type=int,default=40,help='Number of epochs')

    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    return parser.parse_args()


# Load CIFAR-10 Data

def load_cifar10():

    print("Loading CIFAR-10 dataset...")
   
    (x_all,y_all),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

    x_all=x_all/255.0
    x_test=x_test/255.0

    y_all=tf.keras.utils.to_categorical(y_all,10)
    y_test=tf.keras.utils.to_categorical(y_test,10)

    val_size=5000
    x_train,x_val=x_all[:-val_size],x_all[-val_size:]
    y_train,y_val=y_all[:-val_size],y_all[-val_size:]

    
    # CIFAR-10 mean and std for normalization 
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # Normalization
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
        x_val[:,:,:,i]=(x_val[:,:,:,i]-mean[i])/std[i]
    
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Image shape: {x_train.shape[1:]}")
    print(f"Number of classes: 10")
    

    return(x_train,y_train),(x_val,y_val),(x_test,y_test)


# Data Augmentation
def augment_image(image, label):
    # Pad by 4 pixels on each side, then random crop to 32x32
    image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)
    image = tf.image.random_crop(image, [32, 32, 3])
    
    # add brightness
    image=tf.image.random_brightness(image,0.1)
    
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    return image, label


# Build Simple CNN Model

def build_model(num_classes=10):

    inputs = keras.Input(shape=(32, 32, 3))
    x = inputs
    
    # First block: 32 filters
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Second block: 64 filters
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Third block: 128 filters
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model


# Main Training Function

def main():
    args = parse_args()

    print("=" * 60)
    print("CIFAR-10 Training with TensorFlow/Keras")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)


    (x_train,y_train),(x_val,y_val),(x_test,y_test)=load_cifar10()


    # Build model based on args.model
    print(f"\nBuilding {args.model} model...")
    if args.model == "attention56":
        model = ResidualAttentionModel56(num_classes=10,att_type=args.att_type)
    elif args.model == "attention92":
        model = ResidualAttentionModel92(num_classes=10,att_type=args.att_type)
    elif args.model == "attention128":
        model = ResidualAttentionModel128(num_classes=10,att_type=args.att_type)
    elif args.model == "attention164":
        model = ResidualAttentionModel164(num_classes=10,att_type=args.att_type)
    
    ## simple version of attetion models(all with one attention module in each stage)
    elif args.model == "attention56_simple":
        model = ResidualAttentionModel56_simple(num_classes=10,att_type=args.att_type)
    elif args.model == "attention92_simple":
        model = ResidualAttentionModel92_simple(num_classes=10,att_type=args.att_type)
    elif args.model == "attention128_simple":
        model = ResidualAttentionModel128_simple(num_classes=10,att_type=args.att_type)
    elif args.model == "attention164_simple":
        model = ResidualAttentionModel164_simple(num_classes=10,att_type=args.att_type)
    else:
        raise ValueError("Unvalid model!")


    # Compile  
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),   
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="top1_accuracy")]
    )

    # Build the model with input shape so we can count parameters
#     model.build(input_shape=(None, 32, 32, 3))
    _ = model(tf.zeros((1,32,32,3)))
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")

    os.makedirs(args.save_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.save_dir, f'{args.model}_best_model.weights.h5')
    callbacks_list = [
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_top1_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_top1_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_top1_accuracy',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]


    # Train
    train_ds=tf.data.Dataset.from_tensor_slices((x_train,y_train))
    train_ds=train_ds.shuffle(50000)
    train_ds=train_ds.map(augment_image)
    train_ds=train_ds.batch(args.batch_size)

    val_ds=tf.data.Dataset.from_tensor_slices((x_val,y_val))
    val_ds=val_ds.batch(args.batch_size)


    history=model.fit(train_ds,epochs=args.epochs,validation_data=val_ds,callbacks=callbacks_list,verbose=1)


    # Load best weights
    print("\nLoading best model weights...")
    model.load_weights(checkpoint_path)
    
    # Final evaluation   
    print("\nEvaluating on test set...")
    
    y_pred = model.predict(x_test, batch_size=args.batch_size, verbose=0)

    # top-5 accuracy
    y_true_idx = tf.argmax(y_test, axis=1, output_type=tf.int32)
    top5_idx = tf.math.top_k(y_pred, k=5).indices
    hits = tf.reduce_any(tf.equal(top5_idx, tf.reshape(y_true_idx, (-1, 1))), axis=1)
    test_top5_acc = tf.reduce_mean(tf.cast(hits, tf.float32)).numpy()
    test_top5_err = 100 * (1 - test_top5_acc)

    #test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results=model.evaluate(x_test,y_test,verbose=0)
    names=model.metrics_names
    test_loss=results[names.index("loss")]
    #test_acc=results[names.index("accuracy")]
#     test_top1_acc=results[names.index("top1_accuracy")]

    print("metrics_names:", model.metrics_names)
    print("results:", results)
    test_top1_acc = results[1]
    test_top1_err=100*(1-test_top1_acc)

    
   
    print(f"Final Test Accuracy: {test_top1_acc*100:.2f}%")
    print("Test Top-1 Error:",test_top1_err)
    print("Test Top-5 Error:",test_top5_err)
    print("=" * 60)
    print(f"Final Test Loss: {test_loss:.4f}")
    print("=" * 60)
    
    # Print training history summary
    if history.history:
        #train_acc = max(history.history['accuracy'])
        train_top1_acc=max(history.history["top1_accuracy"])
        #val_acc = max(history.history['val_accuracy'])
        val_top1_acc=max(history.history["val_top1_accuracy"])
        train_top1_err=100*(1-max(history.history["top1_accuracy"]))
#         train_top5_err=100*(1-max(history.history["top_5_accuracy"]))
        val_top1_err=100*(1-max(history.history["val_top1_accuracy"]))
#         val_top5_err=100*(1-max(history.history["val_top_5_accuracy"]))
        print("Training Top-1 Error:",train_top1_err)
#         print("Training Top-5 Error:",train_top5_err)
        print("Validation Top-1 Error:",val_top1_err)
#         print("Validation Top-5 Error:",val_top5_err)

        #print(f"\nBest Training Accuracy: {train_acc * 100:.2f}%")
        #print(f"Best Validation Accuracy: {val_acc * 100:.2f}%")
        #print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
        print(f"\nBest Training Accuracy: {train_top1_acc*100:.2f}%")
        print(f"Best Validation Accuracy: {val_top1_acc*100:.2f}%")
        print(f"Final Test Accuracy: {test_top1_acc*100:.2f}%")

    
    print("\nTraining complete!")
    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

