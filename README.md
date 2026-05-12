# Narrative Frame Prediction via Multimodal Deep Learning

## Quick Links

* **Experiments Notebook** – `Final_Assessment_Template_ jathin.ipynb`
* **Model Architectures** – `src/model.py`
* **Training Pipeline** – `src/train.py`
* **Utilities & Preprocessing** – `src/utils.py`
* **Outputs** – `DNN Outputs/`
---
# Project Overview

This project focuses on **multimodal sequence modelling for visual story reasoning** using deep learning.
The objective is to predict the **next multimodal narrative frame** given a sequence of previous image-caption pairs.

The system processes:

* Visual information (images)
* Textual information (descriptions)
* Temporal dependencies across sequential story frames

To improve multimodal alignment, the project introduces a **Cross-Modal Attention mechanism** instead of simple concatenation fusion.

The dataset used is the **StoryReasoning Dataset**:

> Oliveira, D. A. P., & Matos, D. M. (2025).
> *StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding and Grounded Story Generation.*
> arXiv:2505.10292

---

# Objective

Given a sequence of **K image-text pairs**, the model predicts the next pair:

[
(I_1, T_1), (I_2, T_2), ..., (I_K, T_K)
\rightarrow
(I_{K+1}, T_{K+1})
]

The system aims to:

* Learn multimodal representations
* Align visual and textual information
* Model temporal dependencies
* Generate coherent multimodal predictions
* Improve narrative understanding through attention mechanisms

---

# Innovation Summary

| # | Component        | Baseline      | Innovation                       | Justification                                       |
| - | ---------------- | ------------- | -------------------------------- | --------------------------------------------------- |
| 1 | Fusion           | Concatenation | Cross-Modal Attention            | Dynamic alignment between image and text embeddings |
| 2 | Temporal Encoder | Basic LSTM    | Bidirectional Sequence Modelling | Improved temporal understanding                     |
| 3 | Grounding        | None          | CoT Grounding + ROI Alignment    | Better entity consistency                           |
| 4 | Explainability   | None          | Attention Visualisation + XAI    | Improved interpretability                           |

---

# Cross-Modal Attention Fusion

Instead of directly concatenating image and text features, the model applies **cross-modal attention** to dynamically align visual and textual representations.

Advantages:

* Better feature interaction between modalities
* Improved sequence understanding
* Enhanced multimodal coherence
* More stable temporal learning
* Better contextual grounding

The system integrates:

1. Visual Encoder
2. Text Encoder
3. Cross-Modal Attention Fusion
4. Temporal Sequence Model
5. Output Prediction Heads

---

# Model Architecture

```text
frames (N, K, C, H, W)
        │
        ▼
Visual Encoder (ResNet18 / CNN)
        │
        ▼
Visual Embeddings
        │
        ├──────────────────────────┐
        │                          │
        ▼                          ▼
Text Encoder (BERT / LSTM)     ROI Grounding
        │                          │
        ▼                          ▼
Text Embeddings             Entity Alignment
        │                          │
        └──────────┬───────────────┘
                   ▼
        Cross-Modal Attention Fusion
                   ▼
          Bidirectional LSTM / GRU
                   ▼
             Temporal Attention
                   ▼
        ┌──────────────────────┐
        ▼                      ▼
 Image Decoder           Text Decoder
        ▼                      ▼
Predicted Frame      Predicted Caption
```

---

# Experiments

Three experiments were performed to evaluate the effectiveness of the proposed improvements.

---

## Experiment 1 — Baseline Model

### Architecture

* ResNet18 visual encoder
* BERT text encoder
* Concatenation fusion
* Bidirectional LSTM

### Observations

* Higher training loss
* Weaker modality alignment
* Lower prediction accuracy
* Less stable convergence

---

## Experiment 2 — Cross-Modal Attention

### Improvements

* Cross-modal attention fusion introduced
* Dynamic feature interaction
* Better image-text alignment

### Results

* Reduced training loss
* Improved sequence understanding
* Better multimodal coherence
* Higher prediction accuracy

---

## Experiment 3 — Pretrained Attention Model

### Improvements

* Frozen pretrained BERT encoder
* Attention-based multimodal fusion
* Improved temporal consistency

### Results

* Faster convergence
* Improved training stability
* Best overall performance
* Better generalisation

---

# Chain-of-Thought (CoT) Grounding Enhancements

The project additionally integrates **CoT grounding mechanisms** from the StoryReasoning dataset.

## Added Improvements

### 1. ROI Contrastive Learning

Aligns entities across frames using InfoNCE loss.

### 2. Frame-Aware Grounding

Matches visual regions to corresponding frame descriptions.

### 3. Entity Pooling

Maintains consistent entity embeddings across sequences.

### 4. CoT Text Integration

Appends reasoning annotations to frame descriptions.

---

# Training Pipeline

The training process includes:

* Image reconstruction loss
* Context reconstruction loss
* Text generation loss
* Re-identification consistency loss
* Contrastive grounding loss
* Entity consistency regularisation

---

# Key Results

| Metric                        | Result                 |
| ----------------------------- | ---------------------- |
| Baseline Accuracy             | Improved progressively |
| Attention Accuracy            | Higher than baseline   |
| Pretrained Attention Accuracy | Best performance       |
| Training Stability            | Significantly improved |
| Multimodal Coherence          | Stronger alignment     |
| Convergence Speed             | Faster                 |

---

# Explainability (XAI)

The project supports multimodal explainability using:

## 1. Attention Visualisation

Shows temporal importance across narrative frames.

## 2. ROI Grounding Visualisation

Displays grounded entities and bounding boxes.

## 3. Integrated Gradients

Highlights influential image regions.

## 4. Grad-CAM++

Spatial attention maps over visual predictions.

Outputs are saved inside:

```text
outputs/xai/
```

---

# Technologies Used

* Python
* PyTorch
* Transformers (BERT)
* Torchvision
* HuggingFace Datasets
* Matplotlib
* NumPy
* BeautifulSoup
* Google Colab

---

# Repository Structure

```text
DNN-FINAL-TEMPLATE/
│
├── README.md
├── DNN-FINAL-TEMPLATE.ipynb
├── Requirements.txt
├── config.yaml.txt
│
├── src/
│   ├── architecture.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│
├── saved_models/
│   ├── baseline_model.pt
│   ├── attention_model.pt
│   ├── pretrained_attention_model.pt
│   └── text_autoencoder.pth
│
└── checkpoints/
    └── training_checkpoints/
```

---

# Configuration (`config.yaml`)

```yaml
training:
  epochs: 20
  batch_size: 8
  learning_rate: 0.001

model:
  latent_dim: 16
  embedding_dim: 16
  hidden_dim: 16
  num_layers: 1
  dropout: true

dataset:
  image_size:
    height: 60
    width: 125
  context_frames: 4
  max_text_length: 120

cot_grounding:
  use_frame_aware_grounding: true
  use_contrastive_roi: true
  use_entity_pooling: true
  use_cot_text: true

loss_weights:
  lambda_reid: 0.10
  lambda_ground_mse: 0.10
  lambda_contrast: 0.10
  lambda_entity_pool: 0.05
```

---

# Requirements (`requirements.txt`)

```text
torch
torchvision
transformers
datasets
matplotlib
numpy
beautifulsoup4
google-colab
Pillow
scikit-learn
tqdm
pyyaml
```

---

# How to Run

## Install dependencies

```bash
pip install -r requirements.txt
```

## Launch notebook

```bash
jupyter notebook experiments.ipynb
```

## Train model

Run all notebook cells sequentially.

## Load pretrained checkpoint

```python
checkpoint = torch.load("saved_models/pretrained_attention_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
```

---

# Conclusion

This project demonstrates that **Cross-Modal Attention significantly improves multimodal sequence modelling** by enhancing alignment between visual and textual representations.

Compared with baseline fusion methods, the proposed approach achieved:

* Better learning efficiency
* Improved prediction accuracy
* Stronger multimodal coherence
* Faster convergence during training
* Improved grounded reasoning

The experiments validate the effectiveness of:

* Attention-based multimodal fusion
* Temporal sequence learning
* CoT grounding mechanisms
* ROI-based entity alignment

for multimodal narrative reasoning tasks.
