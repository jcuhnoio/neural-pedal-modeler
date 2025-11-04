import torch
from torch.utils.data import Dataset
import numpy as np


class WindowedAudioDataset(Dataset):
    """
    Wraps an audio dataset to provide sliding window context.
    
    Args:
        base_dataset: The underlying dataset
        window_size_input: Number of past input samples to include
        window_size_output: Number of past output samples to include
    """
    
    def __init__(self, base_dataset, window_size_input=0, window_size_output=0):
        self.base_dataset = base_dataset
        self.window_size_input = window_size_input
        self.window_size_output = window_size_output
        self.total_window = max(window_size_input, window_size_output)
        
    def __len__(self):
        # Reduce length by window size to avoid padding
        return len(self.base_dataset) - self.total_window
    
    def __getitem__(self, idx):
        # Adjust index to account for window
        actual_idx = idx + self.total_window
        
        # Get current sample
        current_input, current_output = self.base_dataset[actual_idx]
        
        # Collect input history
        input_history = []
        for i in range(self.window_size_input, 0, -1):
            past_input, _ = self.base_dataset[actual_idx - i]
            input_history.append(past_input)
        
        # Collect output history
        output_history = []
        for i in range(self.window_size_output, 0, -1):
            _, past_output = self.base_dataset[actual_idx - i]
            output_history.append(past_output)
        
        # Concatenate: [past inputs..., current input, past outputs...]
        input_features = input_history + [current_input] + output_history
        
        # Stack along feature dimension
        if isinstance(current_input, torch.Tensor):
            windowed_input = torch.cat(input_features, dim=-1)
        else:
            windowed_input = np.concatenate(input_features, axis=-1)
        
        return windowed_input, current_output
    
    def get_input_dim(self):
        """Calculate the total input dimension after windowing."""
        sample_input, _ = self.base_dataset[0]
        
        if isinstance(sample_input, torch.Tensor):
            base_dim = sample_input.shape[-1]
        else:
            base_dim = sample_input.shape[-1] if len(sample_input.shape) > 0 else 1
        
        total_dim = base_dim * (1 + self.window_size_input + self.window_size_output)
        return total_dim
