#!/usr/bin/env python3
"""
Training script for the pedal modeling neural network.

This script trains a neural network to model guitar pedal effects
and exports the trained model to ONNX format for CLAP plugin integration.
"""

import argparse
import torch
from pathlib import Path

from neural_network.models import PedalNet
from neural_network.utils import create_data_loaders, Trainer
from neural_network.utils.onnx_export import convert_checkpoint_to_onnx
from neural_network.configs.default_config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    DATA_CONFIG,
    EXPORT_CONFIG,
)
from data.windowed_dataset import WindowedAudioDataset
from config import WINDOW_SIZE_INPUT, WINDOW_SIZE_OUTPUT


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a neural network for pedal modeling"
    )
    
    parser.add_argument(
        "--train-input",
        type=str,
        default=DATA_CONFIG["train_input_dir"],
        help="Directory containing training input audio files",
    )
    parser.add_argument(
        "--train-target",
        type=str,
        default=DATA_CONFIG["train_target_dir"],
        help="Directory containing training target audio files",
    )
    parser.add_argument(
        "--val-input",
        type=str,
        default=DATA_CONFIG["val_input_dir"],
        help="Directory containing validation input audio files",
    )
    parser.add_argument(
        "--val-target",
        type=str,
        default=DATA_CONFIG["val_target_dir"],
        help="Directory containing validation target audio files",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG["num_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAINING_CONFIG["batch_size"],
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TRAINING_CONFIG["learning_rate"],
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=EXPORT_CONFIG["checkpoint_dir"],
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the best model to ONNX format after training",
    )
    parser.add_argument(
        "--onnx-output",
        type=str,
        default=None,
        help="Path to save ONNX model (default: models/pedal_model.onnx)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cpu or cuda)",
    )
    
    return parser.parse_args()


def create_dataloaders(args):
    """Create data loaders for training and validation."""
    # Create base datasets
    train_dataset, val_dataset = create_data_loaders(
        train_input_dir=args.train_input,
        train_target_dir=args.train_target,
        val_input_dir=args.val_input,
        val_target_dir=args.val_target,
        batch_size=args.batch_size,
        sequence_length=TRAINING_CONFIG["sequence_length"],
        sample_rate=TRAINING_CONFIG["sample_rate"],
        num_workers=DATA_CONFIG["num_workers"],
    )
    
    # Wrap with windowed dataset if window sizes are configured
    if WINDOW_SIZE_INPUT > 0 or WINDOW_SIZE_OUTPUT > 0:
        train_dataset = WindowedAudioDataset(
            train_dataset,
            window_size_input=WINDOW_SIZE_INPUT,
            window_size_output=WINDOW_SIZE_OUTPUT
        )
        val_dataset = WindowedAudioDataset(
            val_dataset,
            window_size_input=WINDOW_SIZE_INPUT,
            window_size_output=WINDOW_SIZE_OUTPUT
        )
        
        print(f"Using sliding window: input_history={WINDOW_SIZE_INPUT}, output_history={WINDOW_SIZE_OUTPUT}")
        print(f"Total input dimension: {train_dataset.get_input_dim()}")
    
    return train_dataset, val_dataset


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 70)
    print("NEURAL PEDAL MODELER - Training Script")
    print("=" * 70)
    print(f"\nDevice: {args.device}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Create data loaders
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)
    
    train_loader, val_loader = create_dataloaders(args)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\n" + "=" * 70)
    print("Creating Model")
    print("=" * 70)
    
    model = PedalNet(**MODEL_CONFIG)
    print(f"Model architecture: PedalNet")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Train the model
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    trainer.train(num_epochs=args.epochs)
    
    # Export to ONNX if requested
    if args.export_onnx:
        print("\n" + "=" * 70)
        print("Exporting to ONNX")
        print("=" * 70)
        
        if args.onnx_output:
            onnx_path = args.onnx_output
        else:
            onnx_path = Path(EXPORT_CONFIG["onnx_output_dir"]) / EXPORT_CONFIG["model_name"]
        
        best_checkpoint = Path(args.checkpoint_dir) / "best_model.pt"
        
        if best_checkpoint.exists():
            convert_checkpoint_to_onnx(
                checkpoint_path=str(best_checkpoint),
                model_class=PedalNet,
                output_path=str(onnx_path),
                model_kwargs=MODEL_CONFIG,
                device=args.device,
            )
            print(f"\n✓ Model exported to {onnx_path}")
        else:
            print(f"\n✗ Checkpoint not found: {best_checkpoint}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
