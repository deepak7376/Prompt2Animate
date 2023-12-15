import torch
from torch import nn
from torch.nn import functional as F
import math



class SelfAttention(nn.Module):
    def __init__(self, n_head, n_embd):
        super(SelfAttention, self).__init__()
        self.n_head = n_head
        self.n_embd = n_embd

        # Define query, key, and value linear projections for each attention head
        self.query_proj = nn.Linear(n_embd, n_embd * n_head, bias=False)
        self.key_proj = nn.Linear(n_embd, n_embd * n_head, bias=False)
        self.value_proj = nn.Linear(n_embd, n_embd * n_head, bias=False)

        # Final linear projection for the output
        self.out_proj = nn.Linear(n_embd * n_head, n_embd)

    def forward(self, x, causal_mask=False):
        # x: Input tensor of shape (batch_size, sequence_length, n_embd)

        # Linear projections for query, key, and value
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Reshape to have multiple heads
        query = query.view(-1, x.size(1), self.n_head, self.n_embd)
        key = key.view(-1, x.size(1), self.n_head, self.n_embd)
        value = value.view(-1, x.size(1), self.n_head, self.n_embd)

        # Transpose to get dimensions (batch_size, n_head, sequence_length, n_embd)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.n_embd ** 0.5)

        # Apply causal mask (set attention to the future positions to -infinity)
        if causal_mask:
            future_mask = torch.triu(torch.ones_like(scores), diagonal=1)
            scores = scores - 1e9 * future_mask.to(scores.device)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(attn_weights, value)

        # Transpose and reshape to get the final output
        attended_values = attended_values.transpose(1, 2).contiguous().view(-1, x.size(1), self.n_embd * self.n_head)

        # Final linear projection
        output = self.out_proj(attended_values)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=512):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * -(math.log(10000.0) / n_embd))
        pe = torch.zeros(1, max_len, n_embd)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_head, n_embd, ff_hid_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Multi-head self-attention
        self.self_attention = SelfAttention(n_head, n_embd)

        # Layer normalization after self-attention
        self.layernorm_1 = nn.LayerNorm(n_embd)

        # Feedforward neural network
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, ff_hid_dim),
            nn.GELU(),
            nn.Linear(ff_hid_dim, n_embd)
        )

        # Layer normalization after feedforward
        self.layernorm_2 = nn.LayerNorm(n_embd)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, causal_mask=False):
        # Multi-head self-attention
        attention_output = self.self_attention(x, causal_mask=causal_mask)

        # Residual connection and layer normalization
        x = x + self.dropout(attention_output)
        x = self.layernorm_1(x)

        # Feedforward neural network
        ff_output = self.feedforward(x)

        # Residual connection and layer normalization
        x = x + self.dropout(ff_output)
        x = self.layernorm_2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_vocab, n_embd, n_head, ff_hid_dim, n_layers, max_len=512, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # Input embedding layer
        self.embedding = nn.Embedding(n_vocab, n_embd)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(n_embd, max_len=max_len)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(n_head, n_embd, ff_hid_dim, dropout) for _ in range(n_layers)
        ])

    def forward(self, x):
        # Input embedding
        x = self.embedding(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x


if __name__ == "__main__":

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Example parameters
    n_vocab = 49408  # Vocabulary size
    n_embd = 768    # Embedding dimension
    n_token = 77    # Maximum number of tokens in a sequence
    n_head = 12
    n_layers = 12
    ff_hid_dim = 768
    max_len = n_token
    batch = 3

    clip = TransformerEncoder(n_vocab, n_embd, n_head, ff_hid_dim, n_layers,  max_len)

    # Generate some example token indices (batch_size = 3, sequence_length = 10)
    example_tokens = torch.randint(0, n_vocab, (batch, max_len))

    # Forward pass
    result = clip(example_tokens)

    # Print the result
    print("Example Token Indices:")
    print(example_tokens.shape)
    print("\nResulting Embeddings:")
    print(result.shape)
