# -*- coding: utf-8 -*-
"""
TD-Net — CLI entry point.

Usage:
    python main.py                      # Full pipeline
    python main.py --skip-download      # Skip Kaggle download
    python main.py --skip-augment       # Skip offline augmentation
    python main.py --evaluate-only      # Evaluate a saved model
    python main.py --gradcam IMAGE      # Grad-CAM on a specific image
"""

import argparse
import os
import sys
import warnings

# Suppress all python warnings and TensorFlow C++ logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="TD-Net: Tuberculosis Detection with Attention-enhanced MobileNetV2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download (assumes data/ already exists)",
    )
    parser.add_argument(
        "--skip-augment",
        action="store_true",
        help="Skip offline augmentation step",
    )
    parser.add_argument(
        "--no-online-augment",
        action="store_true",
        help="Disable heavy on-the-fly (online) image augmentation during training",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate an existing saved model on the test set",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Run Grad-CAM visualization on 4 Normal and 4 TB images from the test set",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specify a custom dataset directory name within data/ (e.g. 'TBX11K')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset:
        config.set_dataset_name(args.dataset)
        
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    import random
    import numpy as np
    import tensorflow as tf
    print(f"Random seed is configured as {config.RANDOM_SEED}...")
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    # ── Grad-CAM only mode ────────────────────────────────────
    if args.gradcam:
        from src.evaluate import visualize_gradcam
        from src.model import ChannelAttention, SpatialAttention, F1Score

        data_root = os.path.join(config.PROJECT_ROOT, "data")
        if not os.path.isdir(data_root):
            print("No data directory found.")
            return

        dataset_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        
        for ds in dataset_dirs:
            config.set_dataset_name(ds)
            
            if not os.path.exists(config.MODEL_CHECKPOINT):
                print(f"No model checkpoint found for dataset: {ds}. Skipping...")
                continue
                
            print(f"\nLoading model for {ds} from {config.MODEL_CHECKPOINT} ...")
            model = tf.keras.models.load_model(
                config.MODEL_CHECKPOINT,
                custom_objects={
                    "ChannelAttention": ChannelAttention,
                    "SpatialAttention": SpatialAttention,
                    "F1Score": F1Score,
                },
            )
            
            normal_dir = os.path.join(config.TEST_DIR, config.CLASS_NAMES[0])
            tb_dir = os.path.join(config.TEST_DIR, config.CLASS_NAMES[1])
            
            if not os.path.isdir(normal_dir) or not os.path.isdir(tb_dir):
                print(f"Skipping {ds} because test directory not found: {normal_dir} or {tb_dir}")
                continue

            normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            tb_images = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            # Reset random seed for Grad-CAM so we get different images each time
            random.seed()
            selected_normal = random.sample(normal_images, min(20, len(normal_images)))
            selected_tb = random.sample(tb_images, min(20, len(tb_images)))
            
            print(f"Running Grad-CAM on selected images for {ds}...")
            for img_path in selected_normal + selected_tb:
                visualize_gradcam(img_path, model)
        return

    # ── Evaluate-only mode ────────────────────────────────────
    if args.evaluate_only:
        import tensorflow as tf
        from src.dataset import get_generators
        from src.evaluate import evaluate_model
        from src.model import ChannelAttention, SpatialAttention, F1Score

        data_root = os.path.join(config.PROJECT_ROOT, "data")
        if not os.path.isdir(data_root):
            print("No data directory found.")
            return

        dataset_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        
        for ds in dataset_dirs:
            config.set_dataset_name(ds)
            
            if not os.path.exists(config.MODEL_CHECKPOINT):
                print(f"No model checkpoint found for dataset: {ds}. Skipping...")
                continue
                
            print(f"\nLoading model for {ds} from {config.MODEL_CHECKPOINT} ...")
            model = tf.keras.models.load_model(
                config.MODEL_CHECKPOINT,
                custom_objects={
                    "ChannelAttention": ChannelAttention,
                    "SpatialAttention": SpatialAttention,
                    "F1Score": F1Score,
                },
            )
            _, _, test_gen = get_generators()
            evaluate_model(model, test_gen)
        return

    # ── Full pipeline ─────────────────────────────────────────
    from src.dataset import download_dataset, split_dataset, augment_dataset, get_generators
    from src.model import create_td_net
    from src.train import train_model
    from src.evaluate import evaluate_model
    from src.utils import plot_class_distribution, plot_training_history

    # 1. Download
    if not args.skip_download:
        download_dataset()
    else:
        print("Skipping dataset download.")

    # 2. Split
    print("\n── Splitting dataset ──")
    split_dataset()

    # 3. Augment
    if not args.skip_augment:
        print("\n── Augmenting dataset ──")
        augment_dataset()
    else:
        print("Skipping augmentation.")

    # 4. Create generators
    print("\n── Creating data generators ──")
    train_gen, val_gen, test_gen = get_generators(online_augmentation=not args.no_online_augment)

    # 5. Show class distribution
    plot_class_distribution(train_gen, config.CLASS_NAMES)

    # 6. Build model
    print("\n── Building TD-Net ──")
    model = create_td_net()
    # model.summary()

    # 7. Train
    print("\n── Training ──")
    history = train_model(model, train_gen, val_gen)

    # 8. Plot training curves
    plot_training_history(history)

    # 9. Evaluate
    print("\n── Evaluating on test set ──")
    evaluate_model(model, test_gen)

    print("\n✅ Done! Results saved to:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
