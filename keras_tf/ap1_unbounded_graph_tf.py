"""
ap1_unbounded_graph_tf.py
─────────────────────────
Anti-Pattern 1: Unbounded Graph Expansion – TensorFlow GradientTape version.

HOW IT IS INJECTED
──────────────────
Inside the manual GradientTape training loop, the @tf.function-decorated
train_step is RE-DEFINED at the start of every epoch. TensorFlow traces
and permanently stores a new computation graph for each definition.

A uniquely-named tf.Variable (tf_loop_epoch_node_<N>) is also created
per epoch, permanently adding graph nodes that are never freed.

Because no graph reset or clear_session() is ever called between epochs,
all traced sub-graphs and variable nodes pile up across all 50 epochs.

EXPECTED EFFECT
───────────────
• Memory footprint grows with every epoch (50 leaked traces + 50 nodes).
• Retracing overhead increases cumulative wall-clock time.
• Unused graph nodes retain references to model weights, preventing
  timely garbage collection.
"""

import numpy as np
import tensorflow as tf

from common import build_tf_model, evaluate, EPOCHS, BATCH
from run_experiment import run_experiment


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

    for epoch in range(EPOCHS):

        # ❌ AP1 – @tf.function RE-DEFINED every epoch inside the loop.
        #    This is the most common manifestation for GradientTape users:
        #    accidentally placing the decorator or function definition inside
        #    the training loop instead of outside it.
        #    Result: TF traces 50 separate graphs, all held in memory.
        @tf.function
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                loss  = loss_fn(y_batch, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        # ❌ AP1 – Unique tf.Variable created per epoch — permanent graph node.
        tf_loop_node = tf.Variable(
            tf.zeros([]),
            trainable=False,
            name=f"tf_loop_epoch_node_{epoch}",   # unique name → new graph node
        )

        epoch_losses = []
        for x_b, y_b in dataset:
            loss = train_step(x_b, y_b)

            # Reference to tf_loop_node keeps the graph node alive
            tf_loop_node.assign(loss)
            epoch_losses.append(loss.numpy())

        # ❌ AP1 – No graph cleanup between epochs.
        #    Missing:  tf.keras.backend.clear_session()
        #    Missing:  tf.compat.v1.reset_default_graph()
        if (epoch + 1) % 10 == 0:
            print(
                f"    Epoch {epoch + 1:>2d}/{EPOCHS}  "
                f"loss={np.mean(epoch_losses):.4f}  "
                f"[{epoch + 1} @tf.function traces accumulated in graph]"
            )

    acc, prec, rec = evaluate(model, X_test, y_test_raw)
    return model, acc, prec, rec


if __name__ == "__main__":
    run_experiment(train_fn, "AP1 Unbounded Graph Expansion TF GradientTape")
