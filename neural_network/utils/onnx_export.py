"""
ONNX export utilities for model deployment.

This module provides functions to export trained PyTorch models to ONNX format
for integration with the CLAP plugin.
"""

import torch
import torch.onnx
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: int = 1,
    sequence_length: int = 2048,
    opset_version: int = 11,
    verify: bool = True
):
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model
        output_path (str): Path to save the ONNX model
        input_size (int): Input feature size (1 for mono audio)
        sequence_length (int): Length of input sequence for export
        opset_version (int): ONNX opset version
        verify (bool): Whether to verify the exported model
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, sequence_length, input_size)
    
    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting model to ONNX format...")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output path: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    print(f"Model exported successfully to {output_path}")
    
    # Verify the exported model
    if verify:
        verify_onnx_model(output_path, dummy_input)


def verify_onnx_model(model_path: str, test_input: torch.Tensor):
    """
    Verify that the exported ONNX model is valid and produces correct outputs.
    
    Args:
        model_path (str): Path to the ONNX model
        test_input (torch.Tensor): Test input tensor
    """
    print("\nVerifying ONNX model...")
    
    # Check model validity
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Test inference with ONNX Runtime
    try:
        ort_session = ort.InferenceSession(str(model_path))
        
        # Prepare input
        input_numpy = test_input.numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: input_numpy}
        
        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"✓ ONNX Runtime inference successful")
        print(f"  Input shape: {input_numpy.shape}")
        print(f"  Output shape: {ort_outputs[0].shape}")
        
    except Exception as e:
        print(f"✗ ONNX Runtime inference failed: {e}")
        raise


def optimize_onnx_model(model_path: str, output_path: str = None):
    """
    Optimize an ONNX model for better performance.
    
    Args:
        model_path (str): Path to the ONNX model
        output_path (str, optional): Path to save optimized model
    """
    import onnx
    from onnx import optimizer
    
    if output_path is None:
        output_path = str(Path(model_path).with_suffix('.optimized.onnx'))
    
    print(f"\nOptimizing ONNX model...")
    
    # Load model
    onnx_model = onnx.load(model_path)
    
    # Apply optimizations
    passes = [
        'eliminate_deadend',
        'eliminate_identity',
        'eliminate_nop_pad',
        'eliminate_nop_transpose',
        'eliminate_unused_initializer',
        'fuse_consecutive_squeezes',
        'fuse_consecutive_transposes',
        'fuse_transpose_into_gemm',
    ]
    
    optimized_model = optimizer.optimize(onnx_model, passes)
    
    # Save optimized model
    onnx.save(optimized_model, output_path)
    print(f"✓ Optimized model saved to {output_path}")


def convert_checkpoint_to_onnx(
    checkpoint_path: str,
    model_class,
    output_path: str,
    model_kwargs: dict = None,
    device: str = 'cpu'
):
    """
    Convert a PyTorch checkpoint to ONNX format.
    
    Args:
        checkpoint_path (str): Path to PyTorch checkpoint
        model_class: Model class to instantiate
        output_path (str): Path to save ONNX model
        model_kwargs (dict): Keyword arguments for model initialization
        device (str): Device to load model on
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    if model_kwargs is None:
        model_kwargs = {}
    model = model_class(**model_kwargs)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Export to ONNX
    export_to_onnx(model, output_path)
