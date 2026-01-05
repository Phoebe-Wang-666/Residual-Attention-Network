
import argparse
import json
import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.attention92_tf import ResidualAttentionModel92
from models.resnet92_tf import ResNet92


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_cifar10(val_size=5000):
    (x_all, y_all), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_all = x_all.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    y_all = y_all.reshape(-1).astype(np.int64)
    y_test = y_test.reshape(-1).astype(np.int64)

    x_train, x_val = x_all[:-val_size], x_all[-val_size:]
    y_train, y_val = y_all[:-val_size], y_all[-val_size:]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# flip to a random incorrect class with prob noise_ratio.
def add_noise(y, noise_ratio, num_classes=10, seed=0):

    if noise_ratio <= 0.0:
        return y.copy()

    rng = np.random.default_rng(seed)
    y = y.astype(np.int64).copy()
    n = y.shape[0]

    flip = rng.random(n) < noise_ratio
    idx = np.where(flip)[0]
    if idx.size == 0:
        return y

 
    r = rng.integers(0, num_classes - 1, size=idx.size)
    true = y[idx]
    noisy = r + (r >= true).astype(np.int64)
    y[idx] = noisy
    return y


def augment_cifar(x, y):
    # pad 4 -> random crop -> flip
    x = tf.image.resize_with_crop_or_pad(x, 32 + 8, 32 + 8)
    x = tf.image.random_crop(x, size=[32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    return x, y


def make_datasets(x_train, y_train_int, x_val, y_val_int, batch_size, num_classes):
    y_train = tf.one_hot(y_train_int, depth=num_classes, dtype=tf.float32)
    y_val = tf.one_hot(y_val_int, depth=num_classes, dtype=tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(50_000, reshuffle_each_iteration=True)
    train_ds = train_ds.map(augment_cifar, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def build_model(model_name, num_classes=10, att_type="arl"):
    name = model_name.lower()
    if name == "attention92":
        return ResidualAttentionModel92(num_classes=num_classes, att_type=att_type)
    if name == "resnet92":
        return ResNet92(num_classes=num_classes)
    raise ValueError("Invalid Model!")

    
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["attention92", "resnet92"])
    p.add_argument("--att_type", default="arl", choices=["arl", "nal"])
    p.add_argument("--noise_levels", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)


    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--train_subset", type=int, default=20000)   
    p.add_argument("--steps_per_epoch", type=int, default=200)   

    p.add_argument("--base_lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=1e-4)

    p.add_argument("--outdir", type=str, default="robustness_runs")
    p.add_argument("--save_json", type=str, default="robustness_results.json")
    return p.parse_args()


def piecewise_lr(global_step, base_lr, boundaries, values):
    
    lr = values[-1]
    for b, v in zip(boundaries[::-1], values[::-1][1:]):
        lr = tf.where(global_step < b, v, lr)
        
        
    lr = tf.where(global_step < boundaries[0], values[0], lr)
    return lr

# update LR based on step
def train_step(model, optimizer, x, y, global_step, base_lr, boundaries, values):
    
    lr = piecewise_lr(global_step, base_lr, boundaries, values)
    optimizer.learning_rate = lr

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits, from_logits=False))
        if model.losses:
            loss = loss + tf.add_n(model.losses)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    pred = tf.argmax(logits, axis=1)
    true = tf.argmax(y, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))
    return loss, acc, lr


def eval_step(model, x, y):
    logits = model(x, training=False)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits, from_logits=False))
    pred = tf.argmax(logits, axis=1)
    true = tf.argmax(y, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))
    return loss, acc


def evaluate(model, ds):
    losses = []
    accs = []
    for x, y in ds:
        l, a = eval_step(model, x, y)
        losses.append(float(l.numpy()))
        accs.append(float(a.numpy()))
    return float(np.mean(losses)), float(np.mean(accs))


def evaluate_test_int_labels(model, x_test, y_test_int, batch_size, num_classes):
    y_test = tf.one_hot(y_test_int, depth=num_classes, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    _, acc = evaluate(model, ds)
    err = 100.0 * (1.0 - acc)
    return acc, err



def run_one_noise(model_name, att_type, noise_ratio, seed, batch_size, max_iters, base_lr, wd, outdir, train_subset):
    set_global_seed(seed)

    num_classes = 10
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10()
    
    # run on the subset of whole data
    if train_subset and train_subset > 0:
        x_train = x_train[:train_subset]
        y_train = y_train[:train_subset]

    # noisy labels only for training set
    y_train_noisy = add_noise(y_train, noise_ratio, num_classes=num_classes, seed=seed)

    train_ds, val_ds = make_datasets(x_train, y_train_noisy, x_val, y_val, batch_size, num_classes)

    model = build_model(model_name, num_classes=num_classes, att_type=att_type)

    # SGD + Nesterov
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9, nesterov=True)


    boundaries = [64000, 96000]   # steps
    values = [base_lr, base_lr * 0.1, base_lr * 0.01]

    # checkpoint
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, f"{model_name}_noise{noise_ratio:.2f}_seed{seed}.weights.h5")

    best_val_acc = -1.0
    best_step = -1
    global_step = tf.Variable(0, dtype=tf.int64)

    t0 = time.time()
    train_iter = iter(train_ds)


    loss_ma = 0.0
    acc_ma = 0.0
    ma_beta = 0.98

    for step in range(1, max_iters + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_ds)
            x, y = next(train_iter)

        loss, acc, lr = train_step(model, optimizer, x, y, global_step, base_lr, boundaries, values)
        global_step.assign_add(1)

        loss_ma = ma_beta * loss_ma + (1 - ma_beta) * float(loss.numpy())
        acc_ma = ma_beta * acc_ma + (1 - ma_beta) * float(acc.numpy())

        # validate occasionally (to keep runtime reasonable)
        if step % 1000 == 0 or step == max_iters:
            _, val_acc = evaluate(model, val_ds)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_step = step
                model.save_weights(ckpt_path)

            print(f"[{model_name}] noise={noise_ratio:.2f} step={step}/{max_iters} "
                  f"lr={float(lr.numpy()):.5f} train_acc(ma)={acc_ma:.4f} val_acc={val_acc:.4f} best={best_val_acc:.4f}")

    # load best and test
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    test_acc, test_err = evaluate_test_int_labels(model, x_test, y_test, batch_size, num_classes)
    seconds = time.time() - t0

    result = {
        "model": model_name,
        "att_type": att_type if model_name.lower() == "attention92" else None,
        "noise_ratio": float(noise_ratio),
        "seed": int(seed),
        "best_val_acc": float(best_val_acc),
        "best_step": int(best_step),
        "test_acc": float(test_acc),
        "test_err_percent": float(test_err),
        "ckpt_path": ckpt_path,
        "seconds": float(seconds),
        "hparams": {
            "batch_size": int(batch_size),
            "max_iters": int(max_iters),
            "base_lr": float(base_lr),
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay_note": "weight decay usually implemented in model layers via kernel_regularizer",
            "boundaries": boundaries,
            "values": values,
        }
    }
    return result


def print_table_style(results):
    noises = [r["noise_ratio"] for r in results]
    errs = [r["test_err_percent"] for r in results]
    accs = [r["test_acc"] for r in results]

    def fmt_row(name, arr, pct=False):
        if pct:
            vals = " ".join([f"{v*100:5.2f}%" if name == "Test acc" else f"{v:5.2f}%" for v in arr])
        else:
            vals = " ".join([f"{v:5.2f}%" for v in arr])
        return f"{name:<10} {vals}"

    noise_line = "Noise     " + " ".join([f"{n*100:5.0f}%" for n in noises])
    err_line = "Test err  " + " ".join([f"{e:5.2f}%" for e in errs])
    acc_line = "Test acc  " + " ".join([f"{a*100:5.2f}%" for a in accs])
    print("\n" + noise_line)
    print(err_line)
    print(acc_line + "\n")





def main():
    args = parse_args()

    all_results = []
    train_n = args.train_subset if args.train_subset and args.train_subset > 0 else 50000 - 5000  
    full_steps = train_n // args.batch_size
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch and args.steps_per_epoch > 0 else full_steps


    max_iters = args.epochs * steps_per_epoch
    print("train_n =", train_n, "batch_size =", args.batch_size,
          "steps/epoch =", steps_per_epoch, "epochs =", args.epochs,
          "=> max_iters =", max_iters)
    
    for noise in args.noise_levels:
        r = run_one_noise(
            model_name=args.model,
            att_type=args.att_type,
            noise_ratio=noise,
            seed=args.seed,
            batch_size=args.batch_size,
            max_iters=max_iters,
            base_lr=args.base_lr,
            wd=args.wd,
            outdir=args.outdir,
            train_subset=args.train_subset,
        )
        all_results.append(r)

    # print table
    print_table_style(all_results)

    # save model
    out = {"model": args.model, "results": all_results}
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved results to {args.save_json}")


if __name__ == "__main__":
    main()
