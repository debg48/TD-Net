# -*- coding: utf-8 -*-
"""
Evaluation and Grad-CAM visualisation for TD-Net.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import plot_confusion_matrix


# ─────────────────────── Test-Set Metrics ──────────────────────

def evaluate_model(model, test_gen):
    """Compute and print accuracy, F1, precision, recall on a test generator."""
    pred_probs = model.predict(test_gen)
    import random
    
    pred_labels = (pred_probs.flatten() > 0.5).astype("int32")
    true_labels = test_gen.classes

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels)

    print("\n╔══════════════════════════════╗")
    print("║     Test-Set Evaluation      ║")
    print("╠══════════════════════════════╣")
    print(f"║  Accuracy  : {acc:.4f}          ║")
    print(f"║  F1 Score  : {f1:.4f}          ║")
    print(f"║  Precision : {prec:.4f}          ║")
    print(f"║  Recall    : {rec:.4f}          ║")
    print("╚══════════════════════════════╝\n")
    
    # Plot the confusion matrix
    plot_confusion_matrix(true_labels, pred_labels, config.CLASS_NAMES)

    # ─────────────────────── Misclassified Samples ────────────────
    misclassified_idx = np.where(true_labels != pred_labels)[0]
    
    if len(misclassified_idx) > 0:
        clean_name = os.path.basename(config.RESULTS_DIR)
        misc_dir = os.path.join(config.PROJECT_ROOT, "results", "misclassified", clean_name)
        os.makedirs(misc_dir, exist_ok=True)
        
        random.seed()
        num_samples = min(4, len(misclassified_idx))
        selected_idx = random.sample(list(misclassified_idx), num_samples)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
        if num_samples == 1:
            axes = [axes]
            
        for i, idx in enumerate(selected_idx):
            filepath = test_gen.filepaths[idx]
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=config.IMAGE_SIZE)
            
            true_cls = config.CLASS_NAMES[int(true_labels[idx])]
            pred_cls = config.CLASS_NAMES[int(pred_labels[idx])]
            
            axes[i].imshow(img)
            axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls}")
            axes[i].axis("off")
            
        plt.tight_layout()
        fig.subplots_adjust(top=0.85) # Provide extra space at the top for multiline titles
        save_path = os.path.join(misc_dir, "misclassified_samples.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved misclassified samples to: {save_path}")

    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


# ─────────────────────── Grad-CAM ──────────────────────────────

def _get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """Generate a Grad-CAM heatmap for a single image."""
    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    return heatmap


def _overlay_heatmap(heatmap, image, alpha=0.4, colormap_name="viridis"):
    """Overlay a heatmap on the original image."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    cmap = plt.get_cmap(colormap_name)
    colors = cmap(np.arange(256))[:, :3]
    colors = (colors * 255).astype(np.uint8)
    heatmap_colored = colors[heatmap_uint8]

    heatmap_colored = tf.image.resize(
        heatmap_colored, (image.shape[0], image.shape[1])
    ).numpy()

    image_norm = image / 255.0
    heatmap_norm = heatmap_colored / 255.0
    return image_norm * (1 - alpha) + heatmap_norm * alpha


def visualize_gradcam(
    image_path,
    model,
    last_conv_layer_name=None,
    save=True,
):
    """
    Generate and display a Grad-CAM overlay for *image_path*.
    Optionally saves the result to ``results/gradcam_<filename>.png``.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = config.LAST_CONV_LAYER_NAME
        
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=config.IMAGE_SIZE
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_input = np.expand_dims(img_array, axis=0)
    img_input = img_input / 255.0

    heatmap = _get_gradcam_heatmap(model, img_input, last_conv_layer_name)

    img_uint8 = np.uint8(img_array)
    overlay = _overlay_heatmap(heatmap, img_uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title("Original Image")
    axes[0].imshow(img_uint8)
    axes[0].axis("off")

    axes[1].set_title("Grad-CAM Overlay")
    axes[1].imshow(overlay)
    axes[1].axis("off")

    plt.tight_layout()

    if save:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        path = os.path.join(config.RESULTS_DIR, f"gradcam_{basename}.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")

    # plt.show()
    plt.close(fig)
