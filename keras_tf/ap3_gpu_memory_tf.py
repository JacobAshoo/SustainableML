"""
ap3_gpu_memory_tf.py
────────────────────
Anti-Pattern 3: GPU Released Memory Failure – TensorFlow GradientTape version.

HOW IT IS INJECTED
──────────────────
Same strategy as the Keras version, adapted for the manual training loop:

  (a) CHECKPOINT ACCUMULATION every 10 epochs:
      Full model clones accumulated in a list with no del / clear_session.

  (b) GRADIENT TAPE CACHE:
      During a dedicated "gradient analysis" pass every 10 epochs,
      a GradientTape is used outside a @tf.function context to record
      gradients for ALL model weights. The resulting gradient tensors are
      appended to a growing list and never deleted.
      In eager mode, gradient tensors ARE GPU-resident (or pinned CPU
      memory) and are not reclaimed by gc.collect().

  (c) ACTIVATION CACHE:
      Intermediate activations from a sub-model are collected every 10
      epochs and kept in a list.

gc.collect() is called after each accumulation step — it returns without
freeing any GPU/device memory, and clear_session() is never called.

EXPECTED EFFECT
───────────────
• Gradient buffers accumulate: 5 sets of full-model gradients in memory.
• Model clones and activation caches grow alongside.
• GPU memory (or pinned CPU memory on CPU-only machines) is never freed,
  causing progressively worse memory pressure over the 50-epoch run.
"""

import gc

import numpy as np
import tensorflow as tf

from common import build_tf_model, evaluate, EPOCHS, BATCH
from run_experiment import run_experiment


@tf.function
def _train_step_ap3(x_batch, y_batch, model, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        loss  = loss_fn(y_batch, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


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

    # Warm-up: force Adam to allocate slot variables in eager mode before
    # @tf.function traces _train_step_ap3 (fixes Keras 3 singleton tf.Variable error).
    dummy_x = X_train[:1]
    dummy_y = y_cat_train[:1]
    with tf.GradientTape() as tape:
        dummy_preds = model(dummy_x, training=True)
        dummy_loss  = loss_fn(dummy_y, dummy_preds)
    dummy_grads = tape.gradient(dummy_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(dummy_grads, model.trainable_variables))

    # ❌ AP3 – Accumulation containers — strong refs prevent GC
    leaked_checkpoints = []   # model clones
    leaked_gradients   = []   # gradient tensor lists
    leaked_activations = []   # intermediate activation arrays

    for epoch in range(EPOCHS):
        epoch_losses = []
        for x_b, y_b in dataset:
            loss = _train_step_ap3(x_b, y_b, model, optimizer, loss_fn)
            epoch_losses.append(loss.numpy())

        # ── Every 10 epochs: inject memory leak ───────────────────────────
        if (epoch + 1) % 10 == 0:

            # ❌ AP3 (a) – Accumulate model clone without releasing it.
            checkpoint = tf.keras.models.clone_model(model)
            checkpoint.set_weights(model.get_weights())
            leaked_checkpoints.append(checkpoint)           # strong ref → leak

            # ❌ AP3 (b) – Record gradients outside @tf.function (eager mode).
            #    Gradient tensors in eager mode are device-resident objects.
            #    Appending to the list keeps them alive indefinitely.
            sample_x = tf.constant(X_train[:BATCH], dtype=tf.float32)
            sample_y = tf.constant(y_cat_train[:BATCH], dtype=tf.float32)
            with tf.GradientTape() as analysis_tape:
                sample_preds = model(sample_x, training=False)
                sample_loss  = loss_fn(sample_y, sample_preds)
            grad_snapshot = analysis_tape.gradient(
                sample_loss, model.trainable_variables
            )
            leaked_gradients.append(grad_snapshot)          # device tensors kept

            # ❌ AP3 (c) – Collect intermediate activations via sub-model.
            #    Build a sub-model that outputs the penultimate Dense layer.
            #    In Keras 3, Sequential models expose .inputs (plural) after
            #    the first call; use model.inputs[0] instead of model.input.
            feat_model = tf.keras.Model(
                inputs=model.inputs[0],
                outputs=model.layers[-3].output,   # Dense(128) output
            )
            activations = feat_model.predict(
                X_test[:1000], verbose=0
            )    # (1000, 128) float32
            leaked_activations.append(activations)          # never freed

            # ❌ AP3 – gc.collect() only — GPU/device memory NOT freed.
            collected = gc.collect()

            grad_mb  = sum(
                g.numpy().nbytes for g in grad_snapshot if g is not None
            ) / 1e6
            act_mb   = sum(a.nbytes for a in leaked_activations) / 1e6
            n_clones = len(leaked_checkpoints)

            print(
                f"    Epoch {epoch + 1:>2d}/{EPOCHS}  "
                f"loss={np.mean(epoch_losses):.4f}  "
                f"| ⚠️  clones={n_clones}  "
                f"grad_snapshots={len(leaked_gradients)} ({grad_mb:.1f} MB each)  "
                f"act_cache={act_mb:.1f} MB  "
                f"gc freed={collected} objects (GPU NOT freed)"
            )
        else:
            if epoch == 0:
                print(f"    Epoch  1/{EPOCHS}  "
                      f"loss={np.mean(epoch_losses):.4f}")

    acc, prec, rec = evaluate(model, X_test, y_test_raw)
    return model, acc, prec, rec


if __name__ == "__main__":
    run_experiment(train_fn, "AP3 GPU Released Memory Failure TF GradientTape")
