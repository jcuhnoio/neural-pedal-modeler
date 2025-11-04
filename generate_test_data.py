#!/usr/bin/env python3
"""
Generate synthetic test data for the neural pedal modeler.

This script creates simple synthetic audio data to test the training pipeline
without needing real guitar recordings.
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def generate_test_signal(duration: float = 10.0, sample_rate: int = 44100):
    """
    Generate a test audio signal with multiple frequency components.
    
    Args:
        duration (float): Duration in seconds
        sample_rate (int): Sample rate in Hz
        
    Returns:
        np.ndarray: Generated audio signal
    """
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    
    # Create a signal with multiple harmonics (simulating guitar)
    signal = np.zeros(samples)
    
    # Fundamental frequencies (guitar strings: E2, A2, D3, G3, B3, E4)
    fundamentals = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
    
    for i, freq in enumerate(fundamentals):
        # Add fundamental and harmonics
        envelope = np.exp(-3 * t) * np.sin(2 * np.pi * 2 * t)  # Amplitude modulation
        
        signal += 0.3 * envelope * np.sin(2 * np.pi * freq * t)  # Fundamental
        signal += 0.15 * envelope * np.sin(2 * np.pi * freq * 2 * t)  # 2nd harmonic
        signal += 0.075 * envelope * np.sin(2 * np.pi * freq * 3 * t)  # 3rd harmonic
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal.astype(np.float32)


def apply_distortion(signal: np.ndarray, gain: float = 3.0):
    """
    Apply simple distortion effect to simulate a pedal.
    
    Args:
        signal (np.ndarray): Input signal
        gain (float): Distortion gain amount
        
    Returns:
        np.ndarray: Distorted signal
    """
    # Apply gain and tanh saturation (soft clipping)
    distorted = np.tanh(signal * gain)
    
    # Scale output
    distorted = distorted * 0.7
    
    return distorted.astype(np.float32)


def main():
    """Generate synthetic test data."""
    print("=" * 70)
    print("Generating Synthetic Test Data")
    print("=" * 70)
    
    # Configuration
    sample_rate = 44100
    duration = 10.0  # seconds
    num_train_files = 5
    num_val_files = 2
    
    # Create directories
    data_dir = Path(__file__).parent / "data"
    train_input_dir = data_dir / "raw" / "input" / "train"
    train_target_dir = data_dir / "raw" / "target" / "train"
    val_input_dir = data_dir / "raw" / "input" / "val"
    val_target_dir = data_dir / "raw" / "target" / "val"
    
    # Generate training data
    print(f"\nGenerating {num_train_files} training files...")
    for i in range(num_train_files):
        # Generate clean signal
        signal = generate_test_signal(duration, sample_rate)
        
        # Apply distortion
        distorted = apply_distortion(signal, gain=2.0 + i * 0.5)
        
        # Save files
        input_path = train_input_dir / f"test_train_{i+1:03d}.wav"
        target_path = train_target_dir / f"test_train_{i+1:03d}.wav"
        
        sf.write(input_path, signal, sample_rate)
        sf.write(target_path, distorted, sample_rate)
        
        print(f"  Created {input_path.name} and {target_path.name}")
    
    # Generate validation data
    print(f"\nGenerating {num_val_files} validation files...")
    for i in range(num_val_files):
        # Generate clean signal
        signal = generate_test_signal(duration, sample_rate)
        
        # Apply distortion
        distorted = apply_distortion(signal, gain=2.5 + i * 0.5)
        
        # Save files
        input_path = val_input_dir / f"test_val_{i+1:03d}.wav"
        target_path = val_target_dir / f"test_val_{i+1:03d}.wav"
        
        sf.write(input_path, signal, sample_rate)
        sf.write(target_path, distorted, sample_rate)
        
        print(f"  Created {input_path.name} and {target_path.name}")
    
    print("\n" + "=" * 70)
    print("Test Data Generation Complete!")
    print("=" * 70)
    print(f"\nGenerated {num_train_files} training pairs and {num_val_files} validation pairs")
    print(f"Duration: {duration} seconds per file")
    print(f"Sample rate: {sample_rate} Hz")
    print("\nYou can now run training with:")
    print("  python train.py --epochs 50 --export-onnx")


if __name__ == "__main__":
    main()
