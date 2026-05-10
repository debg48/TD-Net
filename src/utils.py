# -*- coding: utf-8 -*-
"""
Plotting utilities for TD-Net.
All figures are saved to the results/ directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _ensure_results_dir():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


def plot_class_distribution(generator, class_names, save=True):
    """Bar chart of class counts from a Keras DirectoryIterator."""
    _ensure_results_dir()

    labels = generator.classes
    counts = np.bincount(labels)

    plt.figure(figsize=(6, 4))
    plt.bar(class_names, counts, color=["#4CAF50", "#F44336"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()

    if save:
        path = os.path.join(config.RESULTS_DIR, "class_distribution.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    # plt.show()
    plt.close()


def plot_training_history(history, save=True):
    """Plot accuracy and loss training curves."""
    _ensure_results_dir()

    acc = history.history["accuracy"]
    loss = history.history["loss"]
    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, acc, label="Training Accuracy")
    if "val_accuracy" in history.history:
        axes[0].plot(epochs, history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, loss, label="Training Loss")
    if "val_loss" in history.history:
        axes[1].plot(epochs, history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()

    if save:
        path = os.path.join(config.RESULTS_DIR, "training_curves.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    # plt.show()
    plt.close(fig)


def plot_batch_images(generator, num_batches=2):
    """Display a few sample batches from a generator."""
    for batch_idx in range(num_batches):
        batch_data, batch_labels = next(iter(generator))
        plt.figure(figsize=(12, 6))
        for i in range(len(batch_data)):
            plt.subplot(1, len(batch_data), i + 1)
            plt.imshow(batch_data[i])
            plt.axis("off")
        plt.suptitle(f"Batch {batch_idx + 1}")
        # plt.show()
        plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save=True):
    """Plot and save a confusion matrix heatmap."""
    _ensure_results_dir()
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save:
        path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    
    # plt.show()
    plt.close()
