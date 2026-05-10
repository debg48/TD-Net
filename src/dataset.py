# -*- coding: utf-8 -*-
"""
Dataset utilities: download, train/val/test split, augmentation,
and Keras ImageDataGenerator creation.
"""

import os
import random
import shutil

import numpy as np
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────── Download ──────────────────────────────

def download_dataset():
    """Download the Kaggle TB dataset using ``opendatasets``."""
    import opendatasets as od

    print(f"Downloading dataset to {config.DATA_DIR} ...")
    od.download(config.DATASET_URL, data_dir=config.DATA_DIR, force=True)
    print("Download complete.")


# ─────────────────────── Train / Val / Test Split ──────────────

def split_dataset():
    """
    Split the raw dataset into train (60%), val (10%), test (30%)
    by copying images into sub-folders.
    """
    if os.path.isdir(config.TRAIN_DIR) and os.listdir(config.TRAIN_DIR):
        print("Dataset already split — skipping.")
        return

    # Create destination folders
    for folder in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        for cls in config.CLASS_NAMES:
            os.makedirs(os.path.join(folder, cls), exist_ok=True)

    # Gather all images
    all_images = []
    for cls in config.CLASS_NAMES:
        cls_dir = os.path.join(config.DATASET_SUBDIR, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Expected class directory not found: {cls_dir}\n"
                "Make sure the dataset is downloaded into data/."
            )
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            if os.path.isfile(fpath):
                all_images.append((fname, cls))

    random.seed(config.RANDOM_SEED)
    random.shuffle(all_images)

    train_end = int(len(all_images) * config.TRAIN_RATIO)
    val_end = train_end + int(len(all_images) * config.VAL_RATIO)

    for idx, (fname, cls) in enumerate(all_images):
        if idx < train_end:
            dest = config.TRAIN_DIR
        elif idx < val_end:
            dest = config.VAL_DIR
        else:
            dest = config.TEST_DIR

        src = os.path.join(config.DATASET_SUBDIR, cls, fname)
        dst = os.path.join(dest, cls, fname)
        shutil.copy(src, dst)

    print(
        f"Split complete — "
        f"train: {train_end}, val: {val_end - train_end}, "
        f"test: {len(all_images) - val_end}"
    )


# ─────────────────────── Augmentation ──────────────────────────

_aug_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest",
)


def _augment_folder(image_dir, label):
    """Generate ``AUGMENT_COUNT`` augmented copies per image in *image_dir*."""
    for fname in os.listdir(image_dir):
        fpath = os.path.join(image_dir, fname)
        if not os.path.isfile(fpath):
            continue
        img = load_img(fpath, target_size=config.IMAGE_SIZE)
        arr = np.expand_dims(img_to_array(img), axis=0)
        count = 0
        for _ in _aug_gen.flow(
            arr,
            batch_size=1,
            save_to_dir=image_dir,
            save_prefix=f"aug_{label}",
            save_format="png",
        ):
            count += 1
            if count >= config.AUGMENT_COUNT:
                break


def augment_dataset():
    """Run offline augmentation on train and val sets."""
    for split_name, split_dir in [("train", config.TRAIN_DIR), ("eval", config.VAL_DIR)]:
        for cls in config.CLASS_NAMES:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                print(f"  Augmenting {split_name}/{cls} ...")
                _augment_folder(cls_dir, cls)
    print("Augmentation complete.")


# ─────────────────────── Generators ────────────────────────────

def get_generators(online_augmentation=True):
    """
    Return ``(train_gen, val_gen, test_gen)`` — Keras
    ``DirectoryIterator`` objects ready for ``model.fit``.
    """
    if online_augmentation:
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode="nearest",
            rescale=1.0 / 255,
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )
    test_gen = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen
