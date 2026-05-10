# TD-Net 🫁 

This repository contains the **official implementation** of **TD-Net**, a novel deep learning architecture that combines **MobileNetV2** with **spatial and channel attention mechanisms** for efficient and accurate **tuberculosis (TB) detection** from chest X-ray images.

The methodology achieves **state-of-the-art performance** with **98% accuracy** and **0.99 AUC-ROC** on benchmark datasets while maintaining a **lightweight architecture** suitable for resource-constrained environments.

---

## 📋 Paper Information



---

## 🎯 Key Highlights

- **98% accuracy** and **0.99 AUC-ROC** on Kaggle TB dataset
- **97% accuracy** and **0.98 AUC-ROC** on TBX11K dataset
- **Lightweight architecture** based on MobileNetV2 (only 12.41 MB)
- **Fast inference time** averaging 1.76 seconds per image
- **Attention mechanisms** for improved feature extraction and interpretability
- **Grad-CAM visualization** for model explainability


---

> [!NOTE]
> Since the implementation was revamped, some experimental variance was introduced. Results obtained using **random seed 42** are the closest to the performance metrics reported in the paper.

---

## 🗂️ Datasets

### Dataset #1: Kaggle TB Chest X-Ray Dataset
- **Source**: Curated by Tawsifur Rahman et al.
- **Total Images**: 4,200 chest X-rays
- **Classes**: 
  - Normal: 3,500 images
  - TB Positive: 700 images
- **Image Format**: PNG (512×512 pixels)
- **Download**: [Kaggle Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

### Dataset #2: TBX11K Dataset
- **Total Images**: 4,600 chest X-rays (for binary classification)
- **Classes**:
  - Healthy: 3,800 images
  - TB Positive: 800 images
- **Image Format**: PNG (512×512 pixels)
- **Download**: [Kaggle TBX11](https://www.kaggle.com/datasets/usmanshams/tbx-11)

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: NVIDIA T4 or better)
- 16GB+ RAM

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/debg48/TD-Net.git
cd TD-Net
```

2. **Create a virtual environment (optional but recommended)**:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

TD-Net has been completely refactored into a modular, production-ready Python package. You can run the entire pipeline—from downloading the data to training and evaluation—using the `main.py` CLI.

### Basic Training Pipeline

To run the complete pipeline (Downloads the default Kaggle dataset -> Splits data -> Augments -> Trains -> Evaluates):

```bash
python main.py
```

### CLI Arguments

You can customize the pipeline using the following flags:

- `--skip-download`: Skips dataset downloading (use if your data is already in the `data/` folder).
- `--skip-augment`: Skips the offline dataset augmentation step.
- `--no-online-augment`: Disables heavy on-the-fly image augmentation during training.
- `--evaluate-only`: Skips training and evaluates an existing saved model.
- `--dataset DATASET`: Specify a custom dataset directory name within `data/` (e.g., `--dataset "TBX11K"`).
- `--gradcam IMAGE`: Run Grad-CAM visualization on a specific image path.

### Using Custom Datasets

To train TD-Net on your own dataset:
1. Place your dataset directory inside the `data/` folder (e.g., `data/my-dataset`).
2. Ensure it has subdirectories for each class (e.g., `data/my-dataset/Normal` and `data/my-dataset/Tuberculosis`).
3. Run the pipeline with your custom dataset name:

```bash
python main.py --skip-download --dataset "my-dataset"
```

### Inference & Grad-CAM

To visualize what the model is focusing on for a specific X-ray image:

```bash
python main.py --gradcam "path/to/chest_xray.png"
```

The resulting heatmap will be saved in the `results/` folder, showing the attention mechanisms at work.

---

## 🔬 Technical Details

### Data Preprocessing
- Image resizing to 224×224×3
- Normalization (rescaling to [0, 1])
- Train/Val/Test split: 60%/10%/30%
- Image augmentation: rotation, flip, shift, zoom, shear

### Training Configuration
- **Optimizer**: RMSprop (learning rate: 0.0001)
- **Loss Function**: Weighted Binary Cross-Entropy
- **Class Weights**: {Normal: 0.94, TB: 1.07}
- **Regularization**: L1 (strength: 0.01), Dropout (0.4)
- **Early Stopping**: Patience of 7 epochs on validation AUC-ROC
- **Hardware**: NVIDIA T4 GPU (15GB VRAM)

### Attention Mechanisms

**Channel Attention**:
```
CA(F) = F · σ(W_avg(AvgPool(F)) + W_max(MaxPool(F)))
```

**Spatial Attention**:
```
SA(F) = F · σ(Conv_7×7(Concat(AvgPool(F), MaxPool(F))))
```

---

## 📈 Visualizations

The repository includes:
- **Training curves**: Accuracy, loss, F1-score, AUC-ROC over epochs
- **Confusion matrices**: For both datasets
- **Grad-CAM heatmaps**: Showing model attention on X-ray images
- **Misclassified samples**: Analysis of failure cases

---

## 🔍 Grad-CAM Analysis

TD-Net uses Gradient-weighted Class Activation Mapping (Grad-CAM) for model interpretability:

- **Before attention modules**: Model fails to focus on relevant lung regions
- **After attention modules**: Model correctly highlights TB-affected areas
- **Clinical relevance**: Enables doctors to verify model decisions

---

## 🏥 Clinical Significance

### Advantages
- **Non-invasive**: Uses standard chest X-rays
- **Fast screening**: Real-time inference for quick diagnosis
- **Resource-efficient**: Deployable in low-resource settings
- **Explainable**: Grad-CAM provides visual explanations
- **High sensitivity**: 99% recall on TB-positive cases

### Limitations
- May struggle with very noisy or low-quality images
- Cannot replace confirmatory sputum tests
- Performance depends on image quality and positioning

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📜 Citation



---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Ms. Ongira Badyopadhyay** (Medical College Kolkata) for medical validation and guidance
- **Tawsifur Rahman et al.** for the Kaggle TB dataset
- **TBX11K dataset creators** for the benchmark dataset

---

## 📧 Contact

For questions, collaborations, or issues:

- **Debgandhar Ghosh**: debgandhar4000@gmail.com
- **Pawan Kumar Singh**: pawansingh.ju@gmail.com

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=debg48/TD-Net&type=Date)](https://star-history.com/#debg48/TD-Net&Date)

---

## 📚 Related Work

Check out our other projects:
- [Retrieval Sensitivity Index](https://github.com/debg48/RAG-metric)
- [JU-LDD](https://github.com/debg48/JU-LDD)

---

**Developed with ❤️ at Jadavpur University, Kolkata**
