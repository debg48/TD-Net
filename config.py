# -*- coding: utf-8 -*-
"""
Centralized configuration for TD-Net.
All hyperparameters, paths, and constants live here.
"""

import os

# ──────────────────────────── Paths ────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Where the raw dataset is downloaded / expected
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_NAME = "tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database"

DATASET_SUBDIR = os.path.join(DATA_DIR, DATASET_NAME)

TRAIN_DIR = os.path.join(DATASET_SUBDIR, "train")
VAL_DIR = os.path.join(DATASET_SUBDIR, "eval")
TEST_DIR = os.path.join(DATASET_SUBDIR, "test")

def set_dataset_name(dataset_name):
    """Dynamically set the dataset directory name."""
    global DATASET_NAME, DATASET_SUBDIR, TRAIN_DIR, VAL_DIR, TEST_DIR, CLASS_NAMES
    global RESULTS_DIR, MODEL_CHECKPOINT
    DATASET_NAME = dataset_name
    DATASET_SUBDIR = os.path.join(DATA_DIR, DATASET_NAME)
    TRAIN_DIR = os.path.join(DATASET_SUBDIR, "train")
    VAL_DIR = os.path.join(DATASET_SUBDIR, "eval")
    TEST_DIR = os.path.join(DATASET_SUBDIR, "test")
    
    clean_name = DATASET_NAME.replace("/", "_").replace(" ", "_")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", clean_name)
    MODEL_CHECKPOINT = os.path.join(RESULTS_DIR, "mobilenet_checkpoint.keras")
    
    if os.path.isdir(DATASET_SUBDIR):
        # Infer class names from subdirectories, ignoring the split folders
        found_dirs = [
            d for d in os.listdir(DATASET_SUBDIR) 
            if os.path.isdir(os.path.join(DATASET_SUBDIR, d)) and d not in ("train", "eval", "test")
        ]
        if len(found_dirs) >= 2:
            CLASS_NAMES = sorted(found_dirs)

# Saved models, plots, Grad-CAM images
_clean_name = DATASET_NAME.replace("/", "_").replace(" ", "_")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", _clean_name)
MODEL_CHECKPOINT = os.path.join(RESULTS_DIR, "mobilenet_checkpoint.keras")

# ──────────────────────────── Dataset ──────────────────────────
DATASET_URL = (
    "https://www.kaggle.com/datasets/tawsifurrahman/"
    "tuberculosis-tb-chest-xray-dataset"
)

CLASS_NAMES = ["Normal", "Tuberculosis"]

# ──────────────────────────── Preprocessing ────────────────────
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4

# Train / Val / Test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 44
AUGMENT_COUNT = 5  # augmented copies per image

# ──────────────────────────── Training ─────────────────────────
EPOCHS = 20
LEARNING_RATE = 1e-4
L1_STRENGTH = 0.01
DROPOUT_RATE = 0.4
DENSE_UNITS = 128

CLASS_WEIGHTS = {0: 0.6, 1: 3.0}

EARLY_STOPPING_PATIENCE = 7

# ──────────────────────────── Grad-CAM ─────────────────────────
LAST_CONV_LAYER_NAME = "Conv_1"  # last conv layer in MobileNetV2
