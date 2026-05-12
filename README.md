# Enhancing Multimodal Sequence Modelling with Cross-Modal Attention

## Quick Links

- **[Experiments Notebook]** – `Final_Assessment_Template_ jathin.ipynb`
- **[Model Architectures]** – `src/model.py`
- **[Training Pipeline]** – `src/train.py`
- **[Utilities & Preprocessing]** – `src/utils.py`
- **[DNN Outputs]** – `DNN Outputs/`

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

Innovation Summary
#	Component	Baseline	Innovation	Justification
1	Fusion	Concatenation	Cross-Modal Attention	Dynamic alignment between image and text embeddings
2	Temporal Model	Basic LSTM	Enhanced Sequence Modelling	Improved learning of temporal dependencies in sequences
3	Modality Alignment	None	Cross-Modal Interaction	Stronger semantic consistency across image-text pairs
4	Feature Learning	Static Features	Attention-Based Representation	Better contextual feature interaction across modalities
5	Prediction Quality	Baseline Output	Improved Multimodal Prediction	More coherent and accurate next-frame generation
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

## Training Pipeline

The training pipeline is designed to learn multimodal representations from sequential image-text data and optimize cross-modal alignment for next-frame prediction.

Start
  │
  ▼
Load Dataset (K image-text sequences)
  │
  ▼
Preprocessing
(Image resize + Text tokenization)
  │
  ▼
Feature Extraction
 ┌──────────────────────┐
 │ Visual Encoder (CNN) │
 │ Text Encoder (BERT)  │
 └──────────────────────┘
  │
  ▼
Cross-Modal Attention Fusion
(Dynamic alignment of image & text features)
  │
  ▼
Temporal Sequence Model
(BiLSTM / GRU)
  │
  ▼
Prediction Heads
 ┌──────────────────────┐
 │ Image Decoder        │
 │ Text Decoder         │
 └──────────────────────┘
  │
  ▼
Loss Computation
(Reconstruction + Alignment + Temporal loss)
  │
  ▼
Backpropagation + Optimization (Adam)
  │
  ▼
Validation & Evaluation
  │
  ▼
Save Best Model Checkpoint
  │
  ▼
End

---

### 1. Data Loading and Preprocessing

* Load sequences of **K image-text pairs** from the dataset
* Resize and normalize images for CNN input
* Tokenize text descriptions using a transformer tokenizer
* Construct fixed-length temporal sequences
* Prepare ground-truth **(K+1)** target image-text pair

---

### 2. Feature Extraction

* **Visual Encoder:** Extract visual embeddings from images using a CNN-based backbone
* **Text Encoder:** Extract contextual text embeddings using a transformer (e.g., BERT)
* Align both modalities into a shared embedding space

---

### 3. Cross-Modal Attention Fusion

* Apply cross-modal attention between visual and textual embeddings
* Learn dynamic interactions between modalities
* Generate fused multimodal representations for each frame

---

### 4. Temporal Sequence Modelling

* Pass fused embeddings through a sequence model (e.g., LSTM/GRU)
* Capture temporal dependencies across the sequence of frames
* Optionally apply bidirectional processing for improved context understanding

---

### 5. Prediction Heads

* **Image Decoder Head:** Predict features of the next image frame
* **Text Decoder Head:** Generate the next caption or description
* Produce final multimodal output for frame **K+1**

---

### 6. Loss Functions

The model is trained using a combination of losses:

* Reconstruction loss (image prediction)
* Text generation loss
* Cross-modal alignment loss
* Temporal consistency loss
* (Optional) contrastive loss for better grounding

---

### 7. Optimization

* Backpropagation through the full multimodal network
* Optimizer: Adam / AdamW
* Learning rate scheduling for stable convergence
* Gradient clipping to avoid exploding gradients

---

### 8. Training Loop

For each epoch:

1. Load batch of sequential samples
2. Encode image and text features
3. Apply cross-modal attention fusion
4. Model temporal dependencies
5. Predict next image-text pair
6. Compute total loss
7. Backpropagate and update weights

---

### 9. Validation

* Evaluate model on unseen sequences
* Measure prediction accuracy and multimodal coherence
* Track training vs validation loss
* Monitor convergence stability

---

### 10. Output

* Trained model checkpoints saved periodically
* Best model selected based on validation performance
* Final model used for multimodal next-frame prediction

---

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
├── config.yaml
│
├── src/
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
data:
dataset_name: "daniel3303/StoryReasoning"
context_frames: 4
image_height: 60
image_width: 125
max_text_len: 120

dataloader:
train_batch_size: 8
val_batch_size: 4
test_batch_size: 4
shuffle_train: true
shuffle_val: true
shuffle_test: false

training:
epochs: 20
learning_rate: 0.001
device: "cuda"
seed: 42

optimizer:
type: "Adam"
lr: 0.001

loss:
image_loss: "L1Loss"
context_loss: "MSELoss"
text_loss: "CrossEntropyLoss"

model:
latent_dim: 16
embedding_dim: 16
gru_hidden_dim: 16
num_layers: 1
dropout: true

text_encoder:
vocab_size: 30522
embedding_dim: 16
hidden_dim: 16
num_layers: 1
dropout: 0.1

text_decoder:
vocab_size: 30522
embedding_dim: 16
hidden_dim: 16
num_layers: 1
dropout: 0.1

visual_encoder:
latent_dim: 16
output_w: 16
output_h: 8

attention:
use_attention: true

cot:
use_frame_aware_grounding: true
use_contrastive_roi: true
use_entity_pooling: true
use_cot_text: true

contrastive:
temperature: 0.07

loss_weights:
lambda_reid: 0.10
lambda_ground_mse: 0.10
lambda_contrast: 0.10
lambda_entity_pool: 0.05

checkpoint:
save_dir: "/content/gdrive/MyDrive/DL_Checkpoints"
text_autoencoder_file: "text_autoencoder.pth"

huggingface:
bert_model: "google-bert/bert-base-uncased"

baseline_experiment:
epochs: 20

attention_experiment:
epochs: 20

pretrained_experiment:
freeze_bert: true
epochs: 20

```

---

# Requirements (`requirements.txt`)

```text
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
datasets>=2.14.0
numpy>=1.24.0
matplotlib>=3.7.0
beautifulsoup4>=4.12.0
Pillow>=9.5.0
pyyaml>=6.0
tqdm>=4.66.0
scikit-learn>=1.3.0
```

---

# How to Run

## Install dependencies

```bash
pip install -r requirements.txt
```

## Launch notebook

```bash
jupyter notebook Final_Assessment_Template_ jathin.ipynb
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
