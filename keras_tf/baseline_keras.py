"""
baseline_keras.py
─────────────────
Baseline experiment – Keras (tf.keras model.fit) pipeline.
No anti-patterns injected.

Training pipeline:
  1. Build compiled Sequential CNN.
  2. Apply standard data augmentation via ImageDataGenerator.
  3. Train with model.fit() for 50 epochs, batch size 32.
  4. Evaluate accuracy / precision / recall on the test set.
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from common import build_keras_model, evaluate, EPOCHS, BATCH
from run_experiment import run_experiment


def train_fn(X_train, y_cat_train, X_test, y_cat_test, y_test_raw):
    model = build_keras_model()

    aug = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    gen   = aug.flow(X_train, y_cat_train, batch_size=BATCH)
    steps = int(np.ceil(len(X_train) / BATCH))

    model.fit(
        gen,
        epochs=EPOCHS,
        steps_per_epoch=steps,
        validation_data=(X_test, y_cat_test),
        verbose=0,
    )

    acc, prec, rec = evaluate(model, X_test, y_test_raw)
    return model, acc, prec, rec


if __name__ == "__main__":
    run_experiment(train_fn, "Baseline Keras")
