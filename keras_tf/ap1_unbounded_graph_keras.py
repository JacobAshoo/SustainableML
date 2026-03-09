"""
ap1_unbounded_graph_keras.py
────────────────────────────
Anti-Pattern 1: Unbounded Graph Expansion – Keras version.

HOW IT IS INJECTED
──────────────────
The baseline model.fit() call is replaced with a manual epoch loop that
uses model.train_on_batch(). The @tf.function-decorated train_step is
RE-DEFINED inside the epoch loop on every iteration, forcing TensorFlow
to trace and store a brand-new computation sub-graph each epoch.

Additionally, a uniquely-named tf.Variable (loop_epoch_var_<N>) is
created per epoch, permanently adding a new graph node on every pass.

Because tf.keras.backend.clear_session() and
tf.compat.v1.reset_default_graph() are NEVER called between epochs,
all traced graphs and variable nodes accumulate in memory for the
entire duration of training.

EXPECTED EFFECT
───────────────
• Memory usage climbs steadily across all 50 epochs.
• Retracing overhead adds latency per epoch.
• After 50 epochs: 50 separate @tf.function traces + 50 tf.Variable
  nodes remain alive — none deallocated.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from common import build_keras_model, evaluate, EPOCHS, BATCH
from run_experiment import run_experiment


def train_fn(X_train, y_cat_train, X_test, y_cat_test, y_test_raw):
    model     = build_keras_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn   = tf.keras.losses.CategoricalCrossentropy()

    aug   = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    gen   = aug.flow(X_train, y_cat_train, batch_size=BATCH)
    steps = int(np.ceil(len(X_train) / BATCH))

    for epoch in range(EPOCHS):

        # ❌ AP1 – @tf.function RE-DEFINED every epoch.
        #    TensorFlow traces a completely new computation graph here.
        #    Without clear_session(), every prior trace remains in memory.
        @tf.function
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                loss  = loss_fn(y_batch, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        # ❌ AP1 – New named tf.Variable created every epoch.
        #    Each call creates a permanent node in the default graph.
        #    The unique name prevents TF from reusing an existing variable.
        epoch_accumulator = tf.Variable(
            tf.zeros([]),
            trainable=False,
            name=f"loop_epoch_var_{epoch}",   # unique → new graph node
        )

        epoch_losses = []
        for step in range(steps):
            x_b, y_b = next(gen)
            x_t = tf.constant(x_b, dtype=tf.float32)
            y_t = tf.constant(y_b, dtype=tf.float32)

            loss_val = train_step(x_t, y_t)

            # Reference keeps epoch_accumulator's graph node alive
            epoch_accumulator.assign(loss_val)
            epoch_losses.append(loss_val.numpy())

        # ❌ AP1 – No graph cleanup here.
        #    Missing:  tf.keras.backend.clear_session()
        #    Missing:  tf.compat.v1.reset_default_graph()
        if (epoch + 1) % 10 == 0:
            print(
                f"    Epoch {epoch + 1:>2d}/{EPOCHS}  "
                f"loss={np.mean(epoch_losses):.4f}  "
                f"[graph has accumulated {epoch + 1} @tf.function traces "
                f"and {epoch + 1} leaked tf.Variable nodes]"
            )

    acc, prec, rec = evaluate(model, X_test, y_test_raw)
    return model, acc, prec, rec


if __name__ == "__main__":
    run_experiment(train_fn, "AP1 Unbounded Graph Expansion Keras")
