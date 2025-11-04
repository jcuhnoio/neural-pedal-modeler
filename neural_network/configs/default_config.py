"""
Default training configuration for the pedal modeling neural network.

This configuration can be loaded and modified for different training runs.
"""

# Model configuration
MODEL_CONFIG = {
    "input_size": 1,
    "hidden_size": 96,
    "num_layers": 2,
    "output_size": 1,
    "dropout": 0.1,
}

# Training configuration
TRAINING_CONFIG = {
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "sequence_length": 8192,  # Audio samples per training sequence
    "sample_rate": 44100,
}

# Data configuration
DATA_CONFIG = {
    "train_input_dir": "data/raw/input/train",
    "train_target_dir": "data/raw/target/train",
    "val_input_dir": "data/raw/input/val",
    "val_target_dir": "data/raw/target/val",
    "num_workers": 4,
}

# Export configuration
EXPORT_CONFIG = {
    "checkpoint_dir": "checkpoints",
    "onnx_output_dir": "models",
    "model_name": "pedal_model.onnx",
    "opset_version": 11,
}
