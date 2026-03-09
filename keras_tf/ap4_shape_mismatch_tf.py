"""
ap4_shape_mismatch_tf.py
────────────────────────
Anti-Pattern 4: Shape Mismatch Leak – TensorFlow GradientTape version.

HOW IT IS INJECTED
──────────────────
The same three-stage padding pipeline from the Keras version is applied
inside the manual GradientTape training loop:

  Stage 1 – PAD every batch to PADDED_SIZE (2×BATCH = 64).
  Stage 2 – Unnecessary float32 → float64 → float32 cast.
  Stage 3 – Append the oversized padded tensor to leaked_batches.

Additionally, at the GradientTape level, model inputs are unnecessarily
flattened (batch, 32, 32, 3) → (batch, 3072) and then immediately
reshaped back before the forward pass. Both the flat and the original
shape are live simultaneously, doubling intermediate tensor memory.

EXPECTED EFFECT
───────────────
• Identical per-epoch leak profile as the Keras version (~1.2 GB/epoch
  before pruning).
• The extra flatten/reshape inside the tape context means the tape
  also records operations on the intermediate flat tensor, slightly
  increasing gradient computation overhead.
"""

import math

import numpy as np
import tensorflow as tf

from common import build_tf_model, evaluate, EPOCHS, BATCH
from run_experiment import run_experiment

PADDED_SIZE = BATCH * 2   # 64


def _pad_and_cast_tf(x_batch: tf.Tensor, y_batch: tf.Tensor):
    """
    Pad batch tensors to PADDED_SIZE, cast through float64, return real slice.
    The full padded tensor is also returned for leak accumulation.
    """
    real_n = tf.shape(x_batch)[0]

    # ❌ AP4 Stage 1 – Oversized zero allocation + scatter update
    padded_x = tf.zeros([PADDED_SIZE, 32, 32, 3], dtype=tf.float32)
    padded_y = tf.zeros([PADDED_SIZE, 10],         dtype=tf.float32)

    indices  = tf.cast(tf.expand_dims(tf.range(real_n), axis=1), tf.int32)
    padded_x = tf.tensor_scatter_nd_update(padded_x, indices, x_batch)
    padded_y = tf.tensor_scatter_nd_update(padded_y, indices, y_batch)

    # Slice to real size — padded_x still fully allocated
    x_real = padded_x[:real_n]
    y_real = padded_y[:real_n]

    # ❌ AP4 Stage 2 – Double-precision cast (peak mem = float32 + float64)
    x_f64  = tf.cast(x_real, tf.float64)
    x_real = tf.cast(x_f64,  tf.float32)

    return x_real, y_real, padded_x   # padded_x is the leaked oversized tensor


def train_fn(X_train, y_cat_train, X_test, y_cat_test, y_test_raw):
    model     = build_tf_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn   = tf.keras.losses.CategoricalCrossentropy()

    dataset = (
        tf.data.Dataset
        .from_tensor_slices((X_train, y_cat_train))
        .shuffle(buffer_size=10_000)
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ❌ AP4 Stage 3 – Growing list of oversized padded tensors
    leaked_batches: list = []

    for epoch in range(EPOCHS):
        epoch_losses = []

        for x_b, y_b in dataset:

            # ❌ AP4 – Pad + cast before entering the tape context
            x_real, y_real, x_padded = _pad_and_cast_tf(x_b, y_b)

            # ❌ AP4 Extra – Flatten and immediately reshape inside tape.
            #    Both shapes are live during gradient computation.
            with tf.GradientTape() as tape:
                flat_x     = tf.reshape(x_real, [tf.shape(x_real)[0], -1])  # (n, 3072)
                x_restored = tf.reshape(flat_x, tf.shape(x_real))           # (n,32,32,3)
                preds = model(x_restored, training=True)
                loss  = loss_fn(y_real, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(loss.numpy())

            # ❌ AP4 – Accumulate full padded tensor (PADDED_SIZE rows, not real_n)
            leaked_batches.append(x_padded)

        # Prune to last 500 to avoid hard OOM on Kaggle free tier
        if len(leaked_batches) > 500:
            leaked_batches = leaked_batches[-500:]

        if (epoch + 1) % 10 == 0:
            leak_mb   = sum(t.numpy().nbytes for t in leaked_batches) / 1e6
            waste_pct = (PADDED_SIZE - BATCH) / PADDED_SIZE * 100
            print(
                f"    Epoch {epoch + 1:>2d}/{EPOCHS}  "
                f"loss={np.mean(epoch_losses):.4f}  "
                f"| ⚠️  padded={PADDED_SIZE} real={BATCH}  "
                f"waste={waste_pct:.0f}%  "
                f"leak_buf={leak_mb:.1f} MB retained"
            )

    acc, prec, rec = evaluate(model, X_test, y_test_raw)
    return model, acc, prec, rec


if __name__ == "__main__":
    run_experiment(train_fn, "AP4 Shape Mismatch Leak TF GradientTape")
