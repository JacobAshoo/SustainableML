"""
ap4_shape_mismatch_keras.py
───────────────────────────
Anti-Pattern 4: Shape Mismatch Leak – Keras version.

HOW IT IS INJECTED
──────────────────
The baseline ImageDataGenerator pipeline is replaced with a manual batch
loop. Before each batch is passed to model.train_on_batch(), it goes
through an unnecessary two-stage reshape pipeline:

  Stage 1 – PAD to oversized tensor:
    The batch (≤32 samples) is scatter-updated into a zero tensor padded
    to the next power of 2 above the epoch's batch count.
    Batch size 32 → padded to 64 (next power of 2 > 32 to be safe for
    variable last-batch sizes).  The padded tensor (64, 32, 32, 3) is
    64 × 32 × 32 × 3 × 4 = 786 432 bytes — twice the needed memory.

  Stage 2 – UNNECESSARY PRECISION CAST:
    The real-sized slice is cast float32 → float64 → float32.
    Peak memory holds both the float32 and float64 versions simultaneously.

  Stage 3 – ACCUMULATE padded tensors:
    The full oversized padded tensor is appended to `leaked_batches`.
    The strong reference prevents the allocator from reclaiming the
    padding rows even though only the first ≤32 rows are ever used.

EXPECTED EFFECT
───────────────
• Per epoch: steps × 786 432 B ≈ 1563 × 786 KB ≈ 1.2 GB of padded
  buffers accumulated in leaked_batches by the end of epoch 1.
• Each subsequent epoch adds another ~1.2 GB to the list.
• The double-precision cast doubles peak per-batch memory at each step.
• Training slows and memory pressure compounds over the 50-epoch run.

NOTE: To prevent the process from running out of memory on Kaggle's
free tier, leaked_batches is pruned to the last 500 entries per epoch
(still large enough to demonstrate the pattern and measurable overhead).
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from common import build_keras_model, evaluate, EPOCHS, BATCH
from run_experiment import run_experiment

# Pad every batch to this fixed oversized bucket.
# 32 is already a power of 2, so we use 2×BATCH to guarantee a mismatch.
PADDED_SIZE = BATCH * 2   # 64 — always larger than the real batch


def _pad_and_cast(x_np: np.ndarray, y_np: np.ndarray):
    """
    Pad a numpy batch to PADDED_SIZE and cast through float64.
    Returns (x_real, y_real, x_padded_leaked) where x_padded_leaked
    retains the over-allocated buffer.
    """
    real_n = x_np.shape[0]

    # ❌ AP4 Stage 1 – Allocate oversized zero tensor, scatter real data.
    padded_x = tf.zeros([PADDED_SIZE, 32, 32, 3], dtype=tf.float32)
    padded_y = tf.zeros([PADDED_SIZE, 10],         dtype=tf.float32)

    x_const  = tf.constant(x_np, dtype=tf.float32)
    y_const  = tf.constant(y_np, dtype=tf.float32)
    indices  = tf.cast(
        tf.expand_dims(tf.range(real_n), axis=1), tf.int32
    )
    padded_x = tf.tensor_scatter_nd_update(padded_x, indices, x_const)
    padded_y = tf.tensor_scatter_nd_update(padded_y, indices, y_const)

    # Slice back to real size (but padded_x still allocated as PADDED_SIZE)
    x_real = padded_x[:real_n]   # shape (real_n, 32, 32, 3)
    y_real = padded_y[:real_n]

    # ❌ AP4 Stage 2 – Unnecessary float64 cast doubles peak per-batch RAM.
    x_f64  = tf.cast(x_real, tf.float64)   # duplicate in float64
    x_real = tf.cast(x_f64,  tf.float32)   # cast back — both versions live briefly

    # Return real-sized tensors for training + full padded tensor for leak
    return x_real, y_real, padded_x   # padded_x shape (PADDED_SIZE, 32,32,3)


def train_fn(X_train, y_cat_train, X_test, y_cat_test, y_test_raw):
    model = build_keras_model()

    aug   = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    gen   = aug.flow(X_train, y_cat_train, batch_size=BATCH)
    steps = int(np.ceil(len(X_train) / BATCH))

    # ❌ AP4 Stage 3 – Container that accumulates oversized padded tensors.
    leaked_batches: list = []

    for epoch in range(EPOCHS):
        epoch_losses = []

        for step in range(steps):
            x_np, y_np = next(gen)

            # ❌ AP4 – Pad + cast every batch; accumulate the padded buffer.
            x_real, y_real, x_padded = _pad_and_cast(x_np, y_np)

            # Only the real-sized slice is used for training
            loss = model.train_on_batch(x_real, y_real)
            if isinstance(loss, (list, tuple)):
                loss = loss[0]
            epoch_losses.append(loss)

            # ❌ AP4 – Strong ref to full padded tensor (never freed)
            leaked_batches.append(x_padded)

        # Prune oldest entries to avoid hard OOM on free-tier Kaggle
        # (still keeps the last 500, demonstrating sustained leakage)
        if len(leaked_batches) > 500:
            leaked_batches = leaked_batches[-500:]

        if (epoch + 1) % 10 == 0:
            leak_mb = sum(
                t.numpy().nbytes for t in leaked_batches
            ) / 1e6
            waste_pct = (PADDED_SIZE - BATCH) / PADDED_SIZE * 100
            print(
                f"    Epoch {epoch + 1:>2d}/{EPOCHS}  "
                f"loss={np.mean(epoch_losses):.4f}  "
                f"| ⚠️  padded_size={PADDED_SIZE} real_batch={BATCH}  "
                f"waste={waste_pct:.0f}%  "
                f"leak_buf={leak_mb:.1f} MB retained"
            )

    acc, prec, rec = evaluate(model, X_test, y_test_raw)
    return model, acc, prec, rec


if __name__ == "__main__":
    run_experiment(train_fn, "AP4 Shape Mismatch Leak Keras")
