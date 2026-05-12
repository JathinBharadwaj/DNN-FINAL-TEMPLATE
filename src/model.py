1. MODEL FILE CODE (architecture.py or model.py)

Visual/Text encoders
class ResNetVisualEncoder(nn.Module)
class TextEncoder(nn.Module)

Fusion modules
class BaselineFusion(nn.Module)
class CrossModalAttention(nn.Module)
class Attention(nn.Module)

Sequence models
class SequenceModel(nn.Module)
class SequencePredictor(nn.Module)

Baseline models
class BaselineModel(nn.Module)
class AttentionModel(nn.Module)

NLP models
class EncoderLSTM(nn.Module)
class DecoderLSTM(nn.Module)
class Seq2SeqLSTM(nn.Module)

Vision autoencoder models
class Backbone(nn.Module)
class AEVisualEncoder(nn.Module)
class AEVisualDecoder(nn.Module)
class AEVisualAutoencoder(nn.Module)

Output heads
class OutputHead(nn.Module)

# A simple attention architecture
"""
Defines an `Attention` module.
It computes attention weights over a sequence of RNN outputs to create a context vector, helping the model focus on relevant parts of the input sequence.
"""

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # This "attention" layer learns a query vector
        self.attn = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1) # Over the sequence length

    def forward(self, rnn_outputs):
        # rnn_outputs shape: [batch, seq_len, hidden_dim]

        # Pass through linear layer to get "energy" scores
        energy = self.attn(rnn_outputs).squeeze(2) # Shape: [batch, seq_len]

        # Get attention weights
        attn_weights = self.softmax(energy) # Shape: [batch, seq_len]

        # Apply weights
        # attn_weights.unsqueeze(1) -> [batch, 1, seq_len]
        # bmm with rnn_outputs -> [batch, 1, hidden_dim]
        context = torch.bmm(attn_weights.unsqueeze(1), rnn_outputs)

        # Squeeze to get final context vector
        return context.squeeze(1) # Shape: [batch, hidden_dim]