Loss functions
text_loss_fn = nn.CrossEntropyLoss()
image_loss_fn = nn.MSELoss()

Compute loss
def compute_loss(...)

Main training function
def train_model(model, dataloader, epochs=20)

Validation function
def validation(model, data_loader)

Plotting results
def plot_results(losses, accuracies, title)

Optimizer setup
optimizer = torch.optim.Adam(...)
ALL training loops

These sections are training code:

baseline_losses, baseline_accs, baseline_time = train_model(...)
attention_losses, attention_accs, attention_time = train_model(...)
pretrained_losses, pretrained_accs, pretrained_time = train_model(...)

Sequence predictor training loop
for epoch in range(N_EPOCHS):
with:
forward pass
backward pass
optimizer.step()
loss calculation

belongs in:
train.py
Text autoencoder training
for epoch in range(N_EPOCHS):
    text_autoencoder.train()

# Training tools
"""
Sets up the optimization process:
1. `criterion_images`: L1 Loss for image reconstruction.
2. `criterion_ctx`: MSE Loss for context guidance (mean color).
3. `criterion_text`: CrossEntropy Loss for text generation.
4. `optimizer`: Adam optimizer for updating model weights.
"""

criterion_images = nn.L1Loss()
criterion_ctx = nn.MSELoss()
criterion_text = nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
optimizer = torch.optim.Adam(sequence_predictor.parameters(), lr=0.001)

Configuration flags to control which Chain-of-Thought (CoT) mechanisms are used during training:
- `USE_FRAME_AWARE_GROUNDING`: Aligns ROI embeddings with the text embedding of the specific frame (vs global).
- `USE_CONTRASTIVE_ROI`: Enables InfoNCE loss to contrast positive ROI-text pairs against negatives.
- `USE_ENTITY_POOLING`: Enforces consistency of embeddings for the same entity within a batch.
- `USE_COT_TEXT`: Appends CoT reasoning text to the input frame description.
"""

# Turn these on/off to control the 4 optional improvements.
USE_FRAME_AWARE_GROUNDING = True      # Option 2: align ROI to matching frame text embedding (instead of always frame 0)
USE_CONTRASTIVE_ROI = True            # Option 1: InfoNCE-style contrastive grounding using batch negatives
USE_ENTITY_POOLING = True             # Option 3: entity-specific pooling/consistency across batch by entity_id
USE_COT_TEXT = True                   # Option 4: concatenate CoT text snippet to the frame descriptions

# Contrastive temperature (only used if USE_CONTRASTIVE_ROI=True)
CONTRASTIVE_TAU = 0.07

