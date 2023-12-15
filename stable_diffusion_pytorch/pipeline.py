from tokenizer import SimpleTokenizer
from clip import TransformerEncoder as CLIP

import torch
from torch import nn
from torch.nn import functional as F
import math


def generate(prompts):
    tokenizer = SimpleTokenizer()
    tokens = tokenizer.tokenize(prompts)
    print(tokens)
    
if __name__ == "__main__":
    prompts = "This is a simple example sentence. Tokenize me!"
    generate(prompts)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Example parameters
    n_vocab = 10000  # Vocabulary size
    n_embd = 512   # Embedding dimension
    n_token = 77    # Maximum number of tokens in a sequence
    n_head = 12
    n_layers = 12
    ff_hid_dim = 768
    max_len = n_token
    batch = 3

    clip = CLIP(n_vocab, n_embd, n_head, ff_hid_dim, n_layers,  max_len)

    # Generate some example token indices (batch_size = 3, sequence_length = 10)
    example_tokens = torch.randint(0, n_vocab, (batch, max_len))

    # Forward pass
    result = clip(example_tokens)

    # Print the result
    print("Example Token Indices:")
    print(example_tokens.shape)
    print("\nResulting Embeddings:")
    print(result.shape)

