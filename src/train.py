# -*- coding: utf-8 -*-
"""
Training loop for TD-Net.
"""

import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm.keras import TqdmCallback

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RESULTS_DIR,
    MODEL_CHECKPOINT,
    EPOCHS,
    LEARNING_RATE,
    CLASS_WEIGHTS,
    EARLY_STOPPING_PATIENCE,
)



def train_model(model, train_gen, val_gen):
    """
    Train *model* and return the ``History`` object.

    Saves the best checkpoint to ``results/mobilenet_checkpoint.keras``.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            MODEL_CHECKPOINT,
            save_best_only=True,
            monitor="accuracy",
        ),
        TqdmCallback(verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=CLASS_WEIGHTS,
        callbacks=callbacks,
        verbose=0,
    )

    print(f"\nBest model saved to: {MODEL_CHECKPOINT}")
    return history
