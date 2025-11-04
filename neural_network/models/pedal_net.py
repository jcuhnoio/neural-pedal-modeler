"""
PedalNet: Neural Network Architecture for Pedal Modeling

This module implements a recurrent neural network architecture optimized
for modeling guitar pedal effects in real-time audio processing.
"""

import torch
import torch.nn as nn


class PedalNet(nn.Module):
    """
    A recurrent neural network for modeling guitar pedal effects.
    
    This architecture uses LSTM layers to capture temporal dependencies
    in audio signals, which is crucial for modeling effects like distortion,
    overdrive, and modulation.
    
    Args:
        input_size (int): Number of input features (typically 1 for mono audio)
        hidden_size (int): Number of hidden units in LSTM layers
        num_layers (int): Number of LSTM layers
        output_size (int): Number of output features (typically 1 for mono audio)
        dropout (float): Dropout probability for regularization
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 96,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1
    ):
        super(PedalNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Input projection layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output projection layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Activation function (Tanh keeps output in [-1, 1] range for audio)
        self.tanh = nn.Tanh()
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            hidden (tuple, optional): Hidden state tuple (h_0, c_0) for LSTM
            
        Returns:
            tuple: (output, hidden_state)
                - output (torch.Tensor): Output tensor of shape (batch_size, sequence_length, output_size)
                - hidden_state (tuple): Hidden state tuple for next sequence
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input to hidden size
        x = self.input_layer(x)
        x = torch.relu(x)
        
        # Pass through LSTM layers
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Project to output size
        output = self.output_layer(lstm_out)
        output = self.tanh(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden state for LSTM.
        
        Args:
            batch_size (int): Batch size
            device (str): Device to create tensors on ('cpu' or 'cuda')
            
        Returns:
            tuple: Initialized hidden state (h_0, c_0)
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)


class WaveNetBlock(nn.Module):
    """
    WaveNet-style convolutional block for audio processing.
    
    This can be used as an alternative architecture for pedal modeling,
    particularly suitable for capturing local temporal patterns.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(WaveNetBlock, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.gate_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.residual_conv = nn.Conv1d(out_channels, in_channels, 1)
        
    def forward(self, x):
        """
        Forward pass through WaveNet block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            tuple: (output, skip_connection)
        """
        residual = x
        
        # Gated activation
        filter_out = torch.tanh(self.conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        x = filter_out * gate_out
        
        # Residual connection
        x = self.residual_conv(x)
        output = x + residual
        
        return output, x
