# DNN-FINAL-TEMPLATE
# Enhancing Multimodal Sequence Modelling with Cross-Modal Attention

# Project Overview
This project focuses on multimodal sequence modelling for visual story reasoning using deep learning. The objective is to predict the next multimodal element in a sequence consisting of images and text descriptions.
The system processes visual and textual information combines and learns temporal dependencies across sequential data. To improve multimodal alignment, the project introduces a Cross-Modal Attention mechanism instead of standard concatenation fusion.

# Objective
Given a sequence of K image-text pairs, the model predicts the next image-text pair (K+1).
The system aims to:
•	Learn multimodal representations
•	Align visual and textual information
•	Model temporal dependencies
•	Generate coherent multimodal predictions
Cross-Modal Attention Fusion
Instead of directly concatenating image and text features, the model applies cross-modal attention to dynamically align visual and textual representations.
•	Better feature interaction between modalities
•	Improved sequence understanding
•	Enhanced multimodal coherence
•	More stable temporal learning
Cross-model attention models use these modals for better performance of the images and textual content
1. Visual Encoder
2. Text Encoder
3. Multimodal Fusion
4. Sequence Model
5. Output Heads
Experiments
Three experiments were performed to evaluate the effectiveness of the proposed improvements.

Experiment 1 — Baseline Model
•	Higher training loss
•	Weaker modality alignment
•	Lower prediction accuracy

Experiment 2 — Cross-Modal Attention
•	Reduced training loss
•	Improved sequence understanding
•	Better multimodal coherence
•	Higher prediction accuracy

Experiment 3 — Pretrained Attention Model
•	Faster convergence
•	Improved training stability
•	Best overall performance

Folder Structure
project/
│	
├── experiment.ipynb
├── README.md
└── results/

# Technologies Used
•	Python
•	PyTorch
•	Transformers (BERT)
•	Torchvision
•	Matplotlib

# Conclusion
This project aims that Cross-Modal Attention improves multimodal sequence modelling by enhancing alignment between visual and textual representations.
Compared with baseline fusion methods, the proposed approach achieved:
•	Better learning efficiency
•	Improved prediction accuracy
•	Stronger multimodal coherence
•	Faster convergence during training
The experiments validate the effectiveness of attention-based fusion for multimodal temporal reasoning tasks.


