"""
ap2_graph_constant_keras.py
───────────────────────────
Anti-Pattern 2: Graph-Constant Bottleneck – Keras version.

HOW IT IS INJECTED
──────────────────
Before training begins, the entire CIFAR-10 training set, test set, and a
large broadcast tensor (mean image repeated 50 000 times) are embedded
directly into the TensorFlow computational graph as tf.constant nodes.

  • X_train_const   shape (50000, 32, 32, 3)  ≈ 192 MB in graph
  • y_train_const   shape (50000, 10)          ≈   2 MB in graph
  • X_test_const    shape (10000, 32, 32, 3)   ≈  38 MB in graph
  • y_test_const    shape (10000, 10)           ≈ 0.4 MB in graph
  • mean_broadcast  shape (50000, 32, 32, 3)   ≈ 192 MB in graph

Training uses a manual epoch loop driven by these graph-resident constants.

EXPECTED EFFECT
───────────────
• ~424 MB of data is permanently pinned inside the TF graph in addition
  to the NumPy copies already in CPU RAM — nearly doubling memory usage.
• The constants are tightly bound to the graph's lifetime; deleting
  Python variables does NOT release the underlying memory.
• Memory is not reclaimed even after training completes, resulting in
  a persistent memory leak for the session's entire duration.
"""

import numpy as np
import tensorflow as tf

from common import build_keras_model, evaluate, EPOCHS, BATCH
from run_experiment import run_experiment


def train_fn(X_train, y_cat_train, X_test, y_cat_test, y_test_raw):
    model     = build_keras_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn   = tf.keras.losses.CategoricalCrossentropy()

    # ❌ AP2 – Entire dataset embedded as permanent tf.constant graph nodes.
    #    These are NOT lazy references — TF allocates and pins the full
    #    arrays in graph memory immediately. They persist for the session.
    X_train_const  = tf.constant(X_train,     dtype=tf.float32,
                                  name="cifar10_X_train_const")
    y_train_const  = tf.constant(y_cat_train, dtype=tf.float32,
                                  name="cifar10_y_train_const")
    X_test_const   = tf.constant(X_test,      dtype=tf.float32,
                                  name="cifar10_X_test_const")
    y_test_const   = tf.constant(y_cat_test,  dtype=tf.float32,
                                  name="cifar10_y_test_const")

    # ❌ AP2 – Large derived constant: per-pixel mean broadcast to full
    #    training-set size. Serves no purpose beyond demonstrating that
    #    large intermediates embedded as constants are never released.
    mean_pixel      = np.mean(X_train, axis=0, keepdims=True)     # (1,32,32,3)
    mean_broadcast  = np.repeat(mean_pixel, len(X_train), axis=0) # (50000,32,32,3)
    mean_const      = tf.constant(mean_broadcast, dtype=tf.float32,
                                   name="cifar10_mean_broadcast_const")
    del mean_broadcast   # NumPy copy freed, but mean_const stays in graph

    n      = X_train_const.shape[0]
    steps  = int(np.ceil(n / BATCH))
    idx    = tf.range(n, dtype=tf.int64)

    print(
        f"    Graph constants pinned: "
        f"X_train={X_train_const.numpy().nbytes / 1e6:.0f} MB, "
        f"mean_broadcast={mean_const.numpy().nbytes / 1e6:.0f} MB"
    )

    for epoch in range(EPOCHS):
        # Shuffle indices (stays in numpy land for simplicity)
        perm = tf.random.shuffle(idx)

        epoch_losses = []
        for step in range(steps):
            batch_idx = perm[step * BATCH : (step + 1) * BATCH]

            # ❌ AP2 – Gather slices from the graph-resident constants.
            #    Each gather op references the pinned constant, keeping
            #    it alive and preventing any memory release.
            x_b = tf.gather(X_train_const, batch_idx)
            y_b = tf.gather(y_train_const, batch_idx)

            # ❌ AP2 – Reference mean_const in every batch computation,
            #    locking it into the active graph indefinitely.
            mean_b = tf.gather(mean_const, batch_idx)
            x_b    = x_b - mean_b + mean_b   # no-op numerically, but keeps ref

            loss_val = model.train_on_batch(x_b, y_b)
            if isinstance(loss_val, (list, tuple)):
                loss_val = loss_val[0]
            epoch_losses.append(loss_val)

        if (epoch + 1) % 10 == 0:
            # Validate using the graph-resident test constants
            val_loss = loss_fn(y_test_const,
                               model(X_test_const, training=False)).numpy()
            print(
                f"    Epoch {epoch + 1:>2d}/{EPOCHS}  "
                f"train_loss={np.mean(epoch_losses):.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"[constants still pinned in graph]"
            )

    # ❌ AP2 – Constants never explicitly released.
    #    Even reassigning the Python names does not free graph nodes.
    acc, prec, rec = evaluate(model, X_test, y_test_raw)
    return model, acc, prec, rec


if __name__ == "__main__":
    run_experiment(train_fn, "AP2 Graph Constant Bottleneck Keras")
