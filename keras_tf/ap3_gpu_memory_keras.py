"""
ap3_gpu_memory_keras.py
───────────────────────
Anti-Pattern 3: GPU Released Memory Failure – Keras version.

HOW IT IS INJECTED
──────────────────
During the 50-epoch training run, the anti-pattern is injected at two
points:

  (a) CHECKPOINT ACCUMULATION every 10 epochs:
      A full model clone (weights + optimizer state) is created via
      tf.keras.models.clone_model() and stored in a Python list.
      The strong reference in the list prevents Python's garbage
      collector from reclaiming either the Python object OR the
      underlying GPU-resident weight tensors.

  (b) ACTIVATION CACHE ACCUMULATION every 10 epochs:
      Model predictions on the full test set are computed and appended
      to a growing cache list. Each entry is a (10000, 10) float32
      array; five entries = 2 MB of cached predictions held in memory
      with no purpose after collection.

After each accumulation, gc.collect() is called. This targets Python
heap objects only — it does NOT flush TensorFlow's CUDA memory allocator.
tf.keras.backend.clear_session() is never called during training.

EXPECTED EFFECT
───────────────
• 5 full model clones remain GPU-resident simultaneously by epoch 50.
• Prediction caches grow across epochs, consuming CPU RAM.
• gc.collect() fails to reclaim GPU memory, causing the CUDA allocator
  to hold progressively more unused buffers.
• On real GPU hardware this pattern eventually triggers OOM errors.
"""

import gc

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from common import build_keras_model, evaluate, EPOCHS, BATCH
from run_experiment import run_experiment


def train_fn(X_train, y_cat_train, X_test, y_cat_test, y_test_raw):
    model = build_keras_model()

    aug   = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    gen   = aug.flow(X_train, y_cat_train, batch_size=BATCH)
    steps = int(np.ceil(len(X_train) / BATCH))

    # ❌ AP3 – Containers that accumulate GPU-resident objects indefinitely.
    leaked_checkpoints   = []   # model clones — weights on GPU/CPU device
    leaked_pred_cache    = []   # prediction tensors — accumulate in CPU RAM
    leaked_weight_copies = []   # raw weight arrays never freed

    optimizer = tf.keras.optimizers.Adam()
    loss_fn   = tf.keras.losses.CategoricalCrossentropy()

    for epoch in range(EPOCHS):
        epoch_losses = []
        for _ in range(steps):
            x_b, y_b = next(gen)
            loss = model.train_on_batch(x_b, y_b)
            if isinstance(loss, (list, tuple)):
                loss = loss[0]
            epoch_losses.append(loss)

        # ── Every 10 epochs: inject memory leak ───────────────────────────
        if (epoch + 1) % 10 == 0:

            # ❌ AP3 (a) – Clone model and accumulate without releasing.
            #    clone_model allocates a full copy of weights on the device.
            #    Appending to the list creates a strong ref → GC cannot free.
            checkpoint = tf.keras.models.clone_model(model)
            checkpoint.set_weights(model.get_weights())   # GPU alloc here
            leaked_checkpoints.append(checkpoint)         # strong ref → leak

            # ❌ AP3 (b) – Accumulate prediction arrays in a growing cache.
            preds = model.predict(X_test, verbose=0)      # (10000,10) float32
            leaked_pred_cache.append(preds)               # never cleared

            # ❌ AP3 (c) – Copy raw weight arrays with no del.
            weight_snapshot = [w.copy() for w in model.get_weights()]
            leaked_weight_copies.append(weight_snapshot)

            # ❌ AP3 – Call gc.collect() only — does NOT free GPU tensors.
            #    Missing: tf.keras.backend.clear_session()
            #    Missing: del checkpoint / del preds / del weight_snapshot
            collected = gc.collect()

            cache_mb = sum(p.nbytes for p in leaked_pred_cache) / 1e6
            n_clones = len(leaked_checkpoints)

            print(
                f"    Epoch {epoch + 1:>2d}/{EPOCHS}  "
                f"loss={np.mean(epoch_losses):.4f}  "
                f"| ⚠️  leaked clones={n_clones}  "
                f"pred_cache={cache_mb:.1f} MB  "
                f"gc freed={collected} objects (GPU memory NOT freed)"
            )
        else:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1:>2d}/{EPOCHS}  "
                      f"loss={np.mean(epoch_losses):.4f}")

    acc, prec, rec = evaluate(model, X_test, y_test_raw)
    return model, acc, prec, rec


if __name__ == "__main__":
    run_experiment(train_fn, "AP3 GPU Released Memory Failure Keras")
