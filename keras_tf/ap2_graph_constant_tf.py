"""
ap2_graph_constant_tf.py
────────────────────────
Anti-Pattern 2: Graph-Constant Bottleneck – TensorFlow GradientTape version.

HOW IT IS INJECTED
──────────────────
Identical constant-embedding approach to the Keras version, but the
training loop uses a manual GradientTape instead of model.train_on_batch.

The full CIFAR-10 dataset and a large broadcast mean tensor are pinned as
tf.constant nodes before training. The tf.data pipeline is then built
directly from these graph-resident constants (from_tensor_slices on a
tf.constant), which locks them into the dataset graph for the entire run.

EXPECTED EFFECT
───────────────
• Same ~424 MB constant bloat as the Keras version.
• tf.data.Dataset built from tf.constant tensors adds another layer of
  graph binding — the pipeline ops hold references to the constants,
  preventing deallocation even if the Python variables are deleted.
"""

import numpy as np
import tensorflow as tf

from common import build_tf_model, evaluate, EPOCHS, BATCH
from run_experiment import run_experiment


@tf.function
def _train_step_ap2(x_batch, y_batch, model, optimizer, loss_fn):
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

    # ❌ AP2 – Full dataset pinned as tf.constant graph nodes.
    X_train_const  = tf.constant(X_train,     dtype=tf.float32,
                                  name="tf_X_train_const")
    y_train_const  = tf.constant(y_cat_train, dtype=tf.float32,
                                  name="tf_y_train_const")
    X_test_const   = tf.constant(X_test,      dtype=tf.float32,
                                  name="tf_X_test_const")
    y_test_const   = tf.constant(y_cat_test,  dtype=tf.float32,
                                  name="tf_y_test_const")

    # ❌ AP2 – Large broadcast mean constant embedded in graph.
    mean_pixel     = np.mean(X_train, axis=0, keepdims=True)
    mean_broadcast = np.repeat(mean_pixel, len(X_train), axis=0)
    mean_const     = tf.constant(mean_broadcast, dtype=tf.float32,
                                  name="tf_mean_broadcast_const")
    del mean_broadcast

    print(
        f"    Graph constants pinned: "
        f"X_train={X_train_const.numpy().nbytes / 1e6:.0f} MB  "
        f"mean_broadcast={mean_const.numpy().nbytes / 1e6:.0f} MB"
    )

    # Force the optimizer to create its slot variables (momentum, velocity)
    # outside of tf.function. Without this, Adam lazily initializes tf.Variables
    # on the first apply_gradients call — which happens *inside* the @tf.function
    # trace, causing "tf.function only supports singleton tf.Variables" errors.
    dummy_x = X_train_const[:1]
    dummy_y = y_train_const[:1]
    with tf.GradientTape() as tape:
        dummy_preds = model(dummy_x, training=True)
        dummy_loss  = loss_fn(dummy_y, dummy_preds)
    dummy_grads = tape.gradient(dummy_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(dummy_grads, model.trainable_variables))

    # ❌ AP2 – tf.data pipeline constructed FROM the graph-resident constants.
    #    from_tensor_slices on a tf.constant binds the constant as a
    #    graph source node — the dataset retains a reference indefinitely.
    dataset = (
        tf.data.Dataset
        .from_tensor_slices((X_train_const, y_train_const))   # ← graph-bound
        .shuffle(buffer_size=10_000)
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )

    for epoch in range(EPOCHS):
        epoch_losses = []
        for x_b, y_b in dataset:

            # ❌ AP2 – Reference mean_const in every batch step to keep
            #    it active as a live graph dependency.
            n_b    = tf.shape(x_b)[0]
            mean_b = mean_const[:n_b]          # slice — keeps const referenced
            x_b    = x_b - mean_b + mean_b     # numeric no-op; graph dep kept

            loss = _train_step_ap2(x_b, y_b, model, optimizer, loss_fn)
            epoch_losses.append(loss.numpy())

        if (epoch + 1) % 10 == 0:
            # Validation using graph-resident test constants
            val_preds = model(X_test_const, training=False)
            val_loss  = loss_fn(y_test_const, val_preds).numpy()
            print(
                f"    Epoch {epoch + 1:>2d}/{EPOCHS}  "
                f"train_loss={np.mean(epoch_losses):.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"[constants remain pinned in graph]"
            )

    acc, prec, rec = evaluate(model, X_test, y_test_raw)
    return model, acc, prec, rec


if __name__ == "__main__":
    run_experiment(train_fn, "AP2 Graph Constant Bottleneck TF GradientTape")
