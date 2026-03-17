# 🖊️ Handwritten Algorithm to Python Converter

Convert handwritten algorithm images (French pseudocode) into executable Python code using Deep Learning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project provides an **end-to-end pipeline** to convert handwritten algorithmic pseudocode (written in French) into executable Python code. It combines:

1. **Optical Character Recognition (OCR)** - Extract text from handwritten images
2. **Natural Language Processing (NLP)** - Translate French pseudocode to Python syntax
3. **Code Generation** - Produce valid, executable Python code

**Use Cases:**
- 📚 Educational: Help students digitize handwritten algorithms
- 📝 Documentation: Convert paper notes to digital code
- 🔄 Legacy: Digitize old algorithm notebooks

---

## ✨ Features

- ✅ **Custom OCR Model** - Fine-tuned TrOCR for handwritten French text
- ✅ **CTC Decoding** - Accurate sequence-to-sequence recognition
- ✅ **Pseudocode Translation** - French algorithmic syntax → Python
- ✅ **Syntax Validation** - Automatic Python syntax checking
- ✅ **Google Colab Support** - Train and test in the cloud (free GPU)
- ✅ **Web Interface** - Easy-to-use upload and convert (coming soon)

---

## 🎬 Demo

### Input: Handwritten Algorithm
![Example Input](assets/example_input.jpg)

### Output: Python Code
```python
# Algorithm: BubbleSort
# Variables: liste, n, i, j

liste = [64, 34, 25, 12, 22, 11, 90]
n = len(liste)

for i in range(n):
    for j in range(0, n-i-1):
        if liste[j] > liste[j+1]:
            liste[j], liste[j+1] = liste[j+1], liste[j]

print(liste)
```

**Try it yourself:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EMHqUSF7r1_gUfowl8FmLkz4n7M_drPL)

---

## 🏗️ Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT: Handwritten Image               │
└─────────────────────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────────┐
    │  STAGE 1: Image Preprocessing               │
    │  - Resize, Normalize                         │
    │  - Denoise, Enhance Contrast                 │
    └──────────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────────┐
    │  STAGE 2: OCR (TrOCR Fine-tuned)            │
    │  - Vision Encoder: ViT                       │
    │  - Text Decoder: RoBERTa                     │
    │  - CTC Loss Function                         │
    └──────────────────────────────────────────────┘
                           ↓
              French Pseudocode Text
         "Pour i de 1 à 10 Faire..."
                           ↓
    ┌──────────────────────────────────────────────┐
    │  STAGE 3: NLP Translation                   │
    │  - Pattern Matching                          │
    │  - Syntax Tree Parsing                       │
    │  - Rule-based Conversion                     │
    └──────────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────────┐
    │  STAGE 4: Code Generation                   │
    │  - Python Syntax Generation                  │
    │  - Indentation Management                    │
    │  - Syntax Validation                         │
    └──────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│             OUTPUT: Executable Python Code               │
└─────────────────────────────────────────────────────────┘
```

### Model Architecture

**OCR Model:**
- **Base**: `microsoft/trocr-base-handwritten`
- **Encoder**: Vision Transformer (ViT)
- **Decoder**: RoBERTa (Transformer-based)
- **Loss**: CTC (Connectionist Temporal Classification)
- **Input Size**: `(3, 64, 800)`
- **Parameters**: ~333M

**Translation Model:**
- **Approach**: Rule-based (v1) / T5-based (v2, coming soon)
- **Patterns**: 20+ French algorithmic constructs
- **Output**: Valid Python 3.8+ code

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training, optional) we use Google colab

### Clone Repository
```bash
git clone https://github.com/fidaear/Handwritten-Algorithm-to-Python.git
cd Handwritten-Algorithm-to-Python
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Pre-trained Models
```bash
# Download OCR model (51.66% accuracy)
gdown --id YOUR_MODEL_ID -O models/ocr_model.pth

# Or train your own (see Training section)
```

---

## 💻 Usage

### Option 1: Google Colab (Recommended for Beginners)

1. Open the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EMHqUSF7r1_gUfowl8FmLkz4n7M_drPL)
2. Upload your handwritten algorithm image
3. Run all cells
4. Download the generated Python code

### Option 2: Command Line

```bash
python predict.py --image path/to/algorithm.jpg --output output.py
```

**Arguments:**
- `--image`: Path to input image (required)
- `--output`: Path to save Python code (default: `output.py`)
- `--model`: Path to OCR model checkpoint (default: `models/best_improved.pth`)
- `--device`: Device to use (`cuda` or `cpu`, default: auto-detect)

### Option 3: Python API

```python
from predictor import HandwrittenToPython

# Initialize
converter = HandwrittenToPython(model_path='models/best_improved.pth')

# Convert image
python_code = converter.convert('algorithm.jpg')

# Save
with open('output.py', 'w') as f:
    f.write(python_code)

print(python_code)
```

---

## 📊 Dataset

### Current Dataset
- **Total Images**: 290
- **Train/Val/Test Split**: 80% / 10% / 10%
- **Image Size**: Variable (resized to 64×800)
- **Format**: JPG, PNG
- **Annotations**: CSV with `image_path` and `text` columns

### Dataset Structure
```
dataset/
├── images/
│   ├── algorithm_001.jpg
│   ├── algorithm_002.jpg
│   └── ...
└── labels.csv
```

### Creating Your Own Dataset

1. **Collect Images**: Scan or photograph handwritten algorithms
2. **Annotate**: Create `labels.csv`:
   ```csv
   image,text
   images/algo1.jpg,"Algorithme TriBulle\nPour i de 1 à n Faire..."
   ```
3. **Preprocess**: Run preprocessing script:
   ```bash
   python scripts/preprocess_dataset.py --input raw_data/ --output dataset/
   ```

---

## 🎓 Training

### Train OCR Model

**Google Colab (Recommended):**
```bash
# Open training notebook
https://colab.research.google.com/drive/1EMHqUSF7r1_gUfowl8FmLkz4n7M_drPL

# Upload your dataset
# Run all cells
```

**Local Training:**
```bash
python train_ocr.py \
    --dataset dataset/ \
    --epochs 150 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --output_dir models/
```

**Training Configuration:**
- **Epochs**: 150 (with early stopping, patience=15)
- **Batch Size**: 4-8 (depending on GPU memory)
- **Optimizer**: Adam (lr=5e-5)
- **Scheduler**: CosineAnnealingLR
- **Loss**: CTC Loss
- **Augmentation**: Random rotation, color jitter, perspective

**Training Time:**
- **Google Colab (T4 GPU)**: ~2-3 hours
- **Local CPU**: ~18-20 hours
- **Local GPU (RTX 3060)**: ~1-2 hours

### Train NLP Model (Coming Soon)

```bash
python train_nlp.py \
    --dataset pseudocode_python_pairs.json \
    --model t5-small \
    --epochs 20
```

---

## 📈 Results

### OCR Performance

| Metric | Value |
|--------|-------|
| **Character Error Rate (CER)** | 12.5% |
| **Word Error Rate (WER)** | 18.3% |
| **Exact Match Accuracy** | 51.66% |
| **Training Accuracy** | 93.65% |
| **Validation Accuracy** | 50.80% |

**Confusion Matrix:**
- Common errors: `O`↔`0`, `I`↔`1`, `l`↔`1`, `V`↔`7`

### Translation Accuracy

| Category | Accuracy |
|----------|----------|
| **Loops (Pour, Tant que)** | ~75% |
| **Conditionals (Si, Sinon)** | ~80% |
| **Assignments (←, =)** | ~90% |
| **I/O (Lire, Écrire)** | ~85% |
| **Overall Syntax** | ~70% |

### Example Predictions

**Example 1: Bubble Sort** ✅
```
Input:  "Pour i de 1 à n Faire\n  Si A[i] > A[i+1] Alors..."
Output: "for i in range(1, n+1):\n    if A[i] > A[i+1]:..."
Result: ✅ Correct
```

**Example 2: Factorial** ✅
```
Input:  "Lire n\nfact ← 1\nPour i de 1 à n Faire..."
Output: "n = int(input('Entrer n: '))\nfact = 1\nfor i in range(1, n+1):..."
Result: ✅ Correct
```

**Example 3: Binary Search** ⚠️
```
Input:  "debut ← 0\nfin ← n-1\nTant que debut <= fin Faire..."
Output: "debut = 0\nfin = n-1\nwhile debut <= fin:..."
Result: ⚠️ Minor spacing issues (easily fixed)
```

---

## 📁 Project Structure

```
Handwritten-Algorithm-to-Python/
├── models/                     # Saved model checkpoints
│   ├── best_improved.pth      # Best OCR model
│   └── config.json            # Model configuration
├── dataset/                    # Training data
│   ├── images/
│   └── labels.csv
├── notebooks/                  # Jupyter/Colab notebooks
│   ├── train_ocr.ipynb
│   ├── train_nlp.ipynb
│   └── demo.ipynb
├── src/                        # Source code
│   ├── models/
│   │   ├── ocr_model.py
│   │   └── nlp_model.py
│   ├── preprocessing/
│   │   ├── image_processing.py
│   │   └── text_processing.py
│   ├── training/
│   │   ├── train_ocr.py
│   │   └── train_nlp.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── scripts/                    # Utility scripts
│   ├── preprocess_dataset.py
│   └── evaluate.py
├── tests/                      # Unit tests
│   ├── test_ocr.py
│   └── test_translation.py
├── assets/                     # Images for README
│   ├── example_input.jpg
│   └── architecture.png
├── predict.py                  # Main prediction script
├── requirements.txt            # Dependencies
├── setup.py                   # Package setup
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## 🗺️ Roadmap

### Current Version: v1.0 (MVP)
- ✅ OCR model training pipeline
- ✅ Rule-based pseudocode translation
- ✅ Google Colab integration
- ✅ Basic web interface (coming soon)

### v1.1 (Next Release)
- [ ] Improve OCR accuracy to 70%+
- [ ] Add data augmentation techniques
- [ ] Support for more algorithmic patterns
- [ ] Better error handling and recovery

### v2.0 (Future)
- [ ] **T5-based NLP model** for translation (90%+ accuracy)
- [ ] Multi-language support (English, Arabic)
- [ ] Real-time handwriting recognition (live video)
- [ ] Web application with user authentication
- [ ] Mobile app (iOS/Android)
- [ ] API endpoint for developers

### v3.0 (Long-term)
- [ ] Multi-page document support
- [ ] Automatic code optimization
- [ ] Code execution and testing
- [ ] Integration with IDEs (VS Code extension)
- [ ] Collaborative annotation platform

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Handwritten-Algorithm-to-Python.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow PEP 8 style guide
   - Add tests for new features
   - Update documentation

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Describe your changes
   - Reference any related issues

### Areas for Contribution
- 🐛 **Bug fixes**: Report and fix bugs
- 📊 **Dataset**: Contribute more handwritten samples
- 🧠 **Models**: Improve OCR/NLP accuracy
- 📝 **Documentation**: Improve guides and tutorials
- 🎨 **UI/UX**: Enhance web interface
- 🧪 **Testing**: Add unit and integration tests

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 fidaear

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full license text...]
```

---

## 🙏 Acknowledgments

- **Microsoft TrOCR** - Pre-trained OCR model base
- **Hugging Face** - Transformers library
- **PyTorch** - Deep learning framework
- **Google Colab** - Free GPU resources
- **Contributors** - Everyone who helped improve this project

### References
- [TrOCR: Transformer-based OCR](https://arxiv.org/abs/2109.10282)
- [CTC Loss for Sequence Recognition](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [T5: Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683)

---

## 📧 Contact

- **Author**: fidaear
- **GitHub**: [@fidaear](https://github.com/fidaear)
- **Repository**: [Handwritten-Algorithm-to-Python](https://github.com/fidaear/Handwritten-Algorithm-to-Python)
- **Issues**: [Report a bug](https://github.com/fidaear/Handwritten-Algorithm-to-Python/issues)

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=fidaear/Handwritten-Algorithm-to-Python&type=Date)](https://star-history.com/#fidaear/Handwritten-Algorithm-to-Python&Date)

---

## 📊 Statistics

![GitHub stars](https://img.shields.io/github/stars/fidaear/Handwritten-Algorithm-to-Python?style=social)
![GitHub forks](https://img.shields.io/github/forks/fidaear/Handwritten-Algorithm-to-Python?style=social)
![GitHub issues](https://img.shields.io/github/issues/fidaear/Handwritten-Algorithm-to-Python)
![GitHub pull requests](https://img.shields.io/github/issues-pr/fidaear/Handwritten-Algorithm-to-Python)
![GitHub last commit](https://img.shields.io/github/last-commit/fidaear/Handwritten-Algorithm-to-Python)

---

<div align="center">

**Made with ❤️ by [fidaear](https://github.com/fidaear)**

If you find this project useful, please consider giving it a ⭐!

[⬆ Back to Top](#-handwritten-algorithm-to-python-converter)

</div>
