"""
Data loading utilities for audio processing.

This module provides classes and functions for loading and preprocessing
audio data for neural network training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import soundfile as sf


class AudioDataset(Dataset):
    """
    Dataset class for loading audio pairs (input/target) for training.
    
    This dataset expects paired audio files where the input is the clean
    guitar signal and the target is the processed signal through the pedal.
    
    Args:
        input_dir (str): Directory containing input audio files
        target_dir (str): Directory containing target audio files
        sequence_length (int): Length of audio sequences for training
        sample_rate (int): Expected sample rate of audio files
    """
    
    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        sequence_length: int = 8192,
        sample_rate: int = 44100
    ):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        
        # Find all audio files
        self.input_files = sorted(list(self.input_dir.glob('*.wav')))
        self.target_files = sorted(list(self.target_dir.glob('*.wav')))
        
        if len(self.input_files) != len(self.target_files):
            raise ValueError(
                f"Mismatch in number of input ({len(self.input_files)}) "
                f"and target ({len(self.target_files)}) files"
            )
        
        # Preload and segment audio data
        self.segments = []
        self._load_and_segment_audio()
        
    def _load_and_segment_audio(self):
        """Load and segment audio files into training sequences."""
        for input_file, target_file in zip(self.input_files, self.target_files):
            # Load audio files
            input_audio, input_sr = sf.read(input_file)
            target_audio, target_sr = sf.read(target_file)
            
            # Verify sample rates match
            if input_sr != self.sample_rate or target_sr != self.sample_rate:
                print(f"Warning: Sample rate mismatch in {input_file.name}")
                continue
            
            # Convert to mono if stereo
            if len(input_audio.shape) > 1:
                input_audio = np.mean(input_audio, axis=1)
            if len(target_audio.shape) > 1:
                target_audio = np.mean(target_audio, axis=1)
            
            # Segment into fixed-length sequences
            num_segments = len(input_audio) // self.sequence_length
            for i in range(num_segments):
                start = i * self.sequence_length
                end = start + self.sequence_length
                
                input_segment = input_audio[start:end]
                target_segment = target_audio[start:end]
                
                self.segments.append({
                    'input': input_segment,
                    'target': target_segment
                })
    
    def __len__(self):
        """Return the number of segments in the dataset."""
        return len(self.segments)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing 'input' and 'target' tensors
        """
        segment = self.segments[idx]
        
        # Convert to PyTorch tensors and add channel dimension
        input_tensor = torch.FloatTensor(segment['input']).unsqueeze(-1)
        target_tensor = torch.FloatTensor(segment['target']).unsqueeze(-1)
        
        return {
            'input': input_tensor,
            'target': target_tensor
        }


def create_data_loaders(
    train_input_dir: str,
    train_target_dir: str,
    val_input_dir: str = None,
    val_target_dir: str = None,
    batch_size: int = 32,
    sequence_length: int = 8192,
    sample_rate: int = 44100,
    num_workers: int = 4
):
    """
    Create training and validation data loaders.
    
    Args:
        train_input_dir (str): Directory with training input audio
        train_target_dir (str): Directory with training target audio
        val_input_dir (str, optional): Directory with validation input audio
        val_target_dir (str, optional): Directory with validation target audio
        batch_size (int): Batch size for training
        sequence_length (int): Length of audio sequences
        sample_rate (int): Audio sample rate
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader) or (train_loader, None) if no validation data
    """
    # Create training dataset
    train_dataset = AudioDataset(
        train_input_dir,
        train_target_dir,
        sequence_length,
        sample_rate
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create validation dataset if directories are provided
    val_loader = None
    if val_input_dir and val_target_dir:
        val_dataset = AudioDataset(
            val_input_dir,
            val_target_dir,
            sequence_length,
            sample_rate
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader
