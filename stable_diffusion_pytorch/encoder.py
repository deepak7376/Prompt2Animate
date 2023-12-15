import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, noise):
        h = F.relu(self.fc1(x))
        mu, log_var = self.fc21(h), self.fc22(h)
        z = self.reparameterize(mu, log_var)
        return z

if __name__ == "__main__":
    # Define parameters
    input_size = 10
    hidden_size = 256
    latent_size = 20

    # Instantiate Encoder
    encoder = Encoder(input_size, hidden_size, latent_size)

    # Generate random input tensor
    random_input = torch.randn(1, input_size)

    # Add noise to input
    noise = torch.randn_like(random_input)  # You can customize the noise as needed

    # Forward pass through the encoder
    encoded_output = encoder(random_input, noise)

    # Print the results
    print("Random Input:")
    print(random_input)

    print("\nEncoded Output:")
    print(encoded_output)