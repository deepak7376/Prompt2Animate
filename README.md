# Stable Diffusion Model using PyTorch

## Overview

This repository contains a PyTorch implementation of a stable diffusion model from scratch. The diffusion model is a powerful probabilistic generative model that has applications in image synthesis, denoising, and other tasks.

## Features

- **PyTorch Implementation:** The diffusion model is implemented using PyTorch, providing flexibility and ease of use for both training and inference.

- **Stability:** The implementation ensures stability during training, allowing for reliable convergence and generation of high-quality samples.

- **Customization:** Easily adapt the model to different datasets and tasks by customizing the architecture and hyperparameters.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch (>=1.7.0)
- Additional dependencies (specified in `requirements.txt`)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/deepak7376/stable_diffusion_pytorch.git
    cd stable_diffusion_pytorch
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained weights
    ```bash
    wget -O saved_models https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
    ```

### Usage

1. Training:

    ```bash
    python train.py --dataset your_dataset --epochs 1000 --lr 0.001
    ```

2. Inference:

    ```bash
    python inference.py --prompt="a dog is flying in the sky" --model_path saved_models/model.pth --num_samples 10
    ```

## Structure

- **`src/`**: Contains the source code for the diffusion model implementation.
- **`data/`**: Placeholder for your dataset or data loading scripts.
- **`saved_models/`**: Directory to store trained model checkpoints.
- **`experiments/`**: Logs and other experiment-related files.

## Contributing

If you find any issues or have suggestions for improvement, feel free to open an issue or create a pull request. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Inspired by the work of [Umar Jamil YouTube Video](https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=12146s).
