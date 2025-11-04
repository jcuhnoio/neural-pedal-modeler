#!/usr/bin/env python3
"""
Inference script for testing trained pedal models.

This script loads a trained model and processes audio files to demonstrate
the pedal modeling effect.
"""

import argparse
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

from neural_network.models import PedalNet
from neural_network.configs import MODEL_CONFIG


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device (str): Device to load model on
        
    Returns:
        nn.Module: Loaded model in eval mode
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = PedalNet(**MODEL_CONFIG)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model


def process_audio(model, input_audio: np.ndarray, chunk_size: int = 8192, device: str = 'cpu'):
    """
    Process audio through the model.
    
    Args:
        model: Trained neural network model
        input_audio (np.ndarray): Input audio array
        chunk_size (int): Size of processing chunks
        device (str): Device for inference
        
    Returns:
        np.ndarray: Processed audio
    """
    output_audio = []
    hidden = None
    
    # Process in chunks
    num_chunks = len(input_audio) // chunk_size
    
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            
            # Get chunk
            chunk = input_audio[start:end]
            
            # Convert to tensor
            chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(-1).to(device)
            
            # Process through model
            output_chunk, hidden = model(chunk_tensor, hidden)
            
            # Convert back to numpy
            output_chunk = output_chunk.squeeze().cpu().numpy()
            output_audio.append(output_chunk)
    
    # Handle remaining samples
    remaining = len(input_audio) % chunk_size
    if remaining > 0:
        chunk = input_audio[-remaining:]
        chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(-1).to(device)
        output_chunk, _ = model(chunk_tensor, hidden)
        output_chunk = output_chunk.squeeze().cpu().numpy()
        output_audio.append(output_chunk)
    
    # Concatenate all chunks
    output_audio = np.concatenate(output_audio)
    
    return output_audio


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference on audio files with trained pedal model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save processed audio",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Processing chunk size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NEURAL PEDAL MODELER - Inference Script")
    print("=" * 70)
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Load input audio
    print(f"\nLoading input audio from {args.input}")
    input_audio, sample_rate = sf.read(args.input)
    
    # Convert to mono if stereo
    if len(input_audio.shape) > 1:
        input_audio = np.mean(input_audio, axis=1)
        print("Converted stereo to mono")
    
    print(f"Input audio shape: {input_audio.shape}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(input_audio) / sample_rate:.2f} seconds")
    
    # Process audio
    print(f"\nProcessing audio (chunk size: {args.chunk_size})...")
    output_audio = process_audio(model, input_audio, args.chunk_size, args.device)
    
    print(f"Output audio shape: {output_audio.shape}")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sf.write(args.output, output_audio, sample_rate)
    print(f"\n✓ Processed audio saved to {args.output}")
    
    print("\n" + "=" * 70)
    print("Inference Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
