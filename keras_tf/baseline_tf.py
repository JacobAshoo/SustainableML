"""
baseline_tf.py
──────────────
Baseline experiment – TensorFlow manual GradientTape training loop.
No anti-patterns injected.

Training pipeline:
  1. Build un-compiled Sequential CNN.
  2. Stream data with tf.data.Dataset (shuffle → batch → prefetch).
  3. Train with a @tf.function-decorated train_step for 50 epochs.
  4. Evaluate accuracy / precision / recall on the test set.
"""

import numpy as np
import tensorflow as tf

from common import build_tf_model, evaluate, EPOCHS, BATCH
from run_experiment import run_experiment


# train_step is defined ONCE outside the training function — correct practice.
@tf.function
def _train_step(x_batch, y_batch, model, optimizer, loss_fn):
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

    # Warm-up: force Adam to create its slot variables (momentum, velocity)
    # in eager mode before @tf.function traces _train_step. Without this,
    # Keras 3 lazily initializes tf.Variables inside the trace, which raises:
    # "tf.function only supports singleton tf.Variables created on the first call."
    dummy_x = X_train[:1]
    dummy_y = y_cat_train[:1]
    with tf.GradientTape() as tape:
        dummy_preds = model(dummy_x, training=True)
        dummy_loss  = loss_fn(dummy_y, dummy_preds)
    dummy_grads = tape.gradient(dummy_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(dummy_grads, model.trainable_variables))

    for epoch in range(EPOCHS):
        epoch_losses = []
        for x_b, y_b in dataset:
            loss = _train_step(x_b, y_b, model, optimizer, loss_fn)
            epoch_losses.append(loss.numpy())

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1:>2d}/{EPOCHS}  "
                  f"mean_loss={np.mean(epoch_losses):.4f}")

    acc, prec, rec = evaluate(model, X_test, y_test_raw)
    return model, acc, prec, rec


if __name__ == "__main__":
    run_experiment(train_fn, "Baseline TF GradientTape")
