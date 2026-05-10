# -*- coding: utf-8 -*-
"""
TD-Net model architecture.

MobileNetV2 backbone with Channel & Spatial Attention for
binary tuberculosis classification from chest X-rays.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Multiply,
    Conv2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IMAGE_SIZE,
    LEARNING_RATE,
    L1_STRENGTH,
    DROPOUT_RATE,
    DENSE_UNITS,
)


# ─────────────────────── Attention Layers ──────────────────────

class ChannelAttention(tf.keras.layers.Layer):
    """
    Channel Attention Module (CBAM-style).
    CA(F) = F · σ(W_avg(AvgPool(F)) + W_max(MaxPool(F)))
    """

    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(1, channels // self.reduction_ratio)

        self.dense_avg_1 = Dense(units=reduced, activation="relu")
        self.dense_avg_2 = Dense(units=channels, activation="sigmoid")
        self.dense_max_1 = Dense(units=reduced, activation="relu")
        self.dense_max_2 = Dense(units=channels, activation="sigmoid")

    def call(self, inputs):
        channel_avg = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        channel_max = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        avg_out = self.dense_avg_2(self.dense_avg_1(channel_avg))
        max_out = self.dense_max_2(self.dense_max_1(channel_max))

        return Multiply()([inputs, avg_out + max_out])

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class SpatialAttention(tf.keras.layers.Layer):
    """
    Spatial Attention Module (CBAM-style).
    SA(F) = F · σ(Conv_7×7(Concat(AvgPool(F), MaxPool(F))))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(
            filters=1, kernel_size=7, padding="same", activation="sigmoid"
        )

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        return Multiply()([inputs, self.conv(concat)])


# ─────────────────────── Custom Metrics ────────────────────────

class F1Score(tf.keras.metrics.Metric):
    """Harmonic mean of precision and recall, computed per-batch."""

    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


# ─────────────────────── Model Builder ─────────────────────────

def _apply_l1_regularization(base_model, l1_strength):
    """Apply L1 regularization to all trainable conv/dense layers."""
    for layer in base_model.layers:
        if hasattr(layer, "kernel_regularizer"):
            layer.kernel_regularizer = tf.keras.regularizers.l1(l1_strength)


def create_td_net(
    input_shape=(*IMAGE_SIZE, 3),
    l1_strength=L1_STRENGTH,
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE,
    dense_units=DENSE_UNITS,
):
    """
    Build and compile the TD-Net model.

    Architecture:
        MobileNetV2 (ImageNet) → Channel Attention → Spatial Attention
        → GAP → Dense(128, ReLU) → Dropout(0.4) → Dense(1, Sigmoid)

    Returns:
        Compiled ``tf.keras.Model``.
    """
    inputs = Input(shape=input_shape)
    base_model = MobileNetV2(
        include_top=False, weights="imagenet", input_tensor=inputs
    )
    _apply_l1_regularization(base_model, l1_strength)

    x = base_model.output
    x = ChannelAttention()(x)
    x = SpatialAttention()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TD_Net")
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            F1Score(),
        ],
    )
    return model
