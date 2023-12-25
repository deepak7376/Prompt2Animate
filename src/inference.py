import os
import argparse
import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Run the inference script.")
    parser.add_argument("--prompt", type=str, help="Text prompt for text-to-image generation")
    parser.add_argument("--uncond-prompt", type=str, default="", help="Unconditional prompt for text-to-image generation (negative prompt)")
    parser.add_argument("--image-path", type=str, default=None, help="Path to the input image for image-to-image generation")
    parser.add_argument("--output-path", type=str, default="output_image.jpg", help="Path to save the output image")
    parser.add_argument("--strength", type=float, default=0.9, help="Strength parameter for image-to-image generation")
    parser.add_argument("--do-cfg", default=True, help="Enable conditional configuration for text-to-image generation")
    parser.add_argument("--cfg-scale", type=float, default=8, help="Scale parameter for conditional configuration")
    parser.add_argument("--sampler", type=str, default="ddpm", help="Sampler name for image-to-image generation")
    parser.add_argument("--num-inference-steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation")
    return parser.parse_args()

def download_model(model_url, save_path):
    # Download the model using wget
    subprocess.run(["wget", "-O", save_path, model_url])

def main():
    args = parse_args()

    DEVICE = "cpu"

    ALLOW_CUDA = False
    ALLOW_MPS = False

    if torch.cuda.is_available() and ALLOW_CUDA:
        DEVICE = "cuda"
    elif (torch.torch.backends.mps.is_built() or torch.backends.mps.is_available()) and ALLOW_MPS:
        DEVICE = "mps"
    print(f"Using device: {DEVICE}")

    # Get the absolute path to the vocabulary and merges files
    vocab_file = os.path.join(os.getcwd(), "data/vocab.json")
    merges_file = os.path.join(os.getcwd(), "data/merges.txt")
    tokenizer = CLIPTokenizer(vocab_file, merges_file=merges_file)

    # Check if the model file exists
    model_file_path = os.path.join(os.getcwd(), "saved_models/v1-5-pruned-emaonly.ckpt")
    if not os.path.exists(model_file_path):
        print(f"Model file '{model_file_path}' not found. Downloading...")
        model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
        download_model(model_url, model_file_path)
        print("Download complete.")

    models = model_loader.preload_models_from_standard_weights(model_file_path, DEVICE)

    if args.image_path:
        input_image = Image.open(args.image_path)
    else:
        input_image = None

    output_image = pipeline.generate(
        prompt=args.prompt,
        uncond_prompt=args.uncond_prompt,
        input_image=input_image,
        strength=args.strength,
        do_cfg=args.do_cfg,
        cfg_scale=args.cfg_scale,
        sampler_name=args.sampler,
        n_inference_steps=args.num_inference_steps,
        seed=args.seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    Image.fromarray(output_image).save(args.output_path)

if __name__ == "__main__":
    main()
