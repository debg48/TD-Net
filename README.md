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

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download datasets**:
```bash
# Using opendatasets
pip install opendatasets
```

Then run the dataset download cells in the notebook, or manually download from the links provided above.

---

## 🚀 Usage

### Training the Model

1. **Open the Jupyter notebook**:
```bash
jupyter notebook TD-Net.ipynb
```

2. **Configure parameters** (in the notebook):
```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.0001
```

3. **Run all cells** to:
   - Load and preprocess data
   - Apply image augmentation
   - Train the TD-Net model
   - Evaluate on test sets
   - Generate visualizations

### Inference on New Images
```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model('models/mobilenet_checkpoint.keras')

# Load and preprocess image
img_path = 'path/to/chest_xray.png'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Make prediction
prediction = model.predict(img_array)
result = "TB Positive" if prediction[0][0] > 0.5 else "Normal"
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

print(f"Prediction: {result} (Confidence: {confidence:.2%})")
```

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
- [Transformer Models for Twitter Sentiment Analysis](https://github.com/yourusername/twitter-sentiment-transformers)
- [Medical Image Classification with Vision Transformers](https://github.com/yourusername/medical-vit)

---

**Developed with ❤️ at Jadavpur University, Kolkata**
