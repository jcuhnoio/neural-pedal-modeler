# Neural Pedal Modeler - Architecture Documentation

## Overview

The Neural Pedal Modeler is a machine learning system designed to capture and replicate the sound characteristics of guitar pedals using neural networks. The system consists of two main components:

1. **Neural Network Training Pipeline** (Python/PyTorch)
2. **DSP Plugin** (CLAP Framework - to be implemented)

The neural network component trains models on paired audio data (clean guitar input and pedal-processed output) and exports them to ONNX format for real-time inference in the CLAP plugin.

## Project Structure

```
neural-pedal-modeler/
├── neural_network/           # Neural network training code
│   ├── __init__.py
│   ├── models/               # Model architectures
│   │   ├── __init__.py
│   │   └── pedal_net.py     # LSTM-based pedal modeling network
│   ├── utils/                # Training utilities
│   │   ├── __init__.py
│   │   ├── data_loader.py   # Audio data loading and preprocessing
│   │   ├── trainer.py       # Training loop and model management
│   │   └── onnx_export.py   # ONNX export utilities
│   └── configs/              # Configuration files
│       └── default_config.py # Default training parameters
├── data/                     # Training data directory
│   ├── raw/                  # Raw audio files
│   │   ├── input/            # Clean guitar signals
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── target/           # Pedal-processed signals
│   │       ├── train/
│   │       └── val/
│   └── processed/            # Preprocessed data (optional)
├── models/                   # Exported ONNX models
├── checkpoints/              # Training checkpoints
├── train.py                  # Main training script
├── requirements.txt          # Python dependencies
└── README.md                 # Project README
```

## Key Components

### 1. Neural Network Architecture (PedalNet)

**File:** `neural_network/models/pedal_net.py`

The `PedalNet` class implements a recurrent neural network (RNN) architecture optimized for audio signal processing:

#### Architecture Details:
- **Input Layer**: Linear projection from input features to hidden size
- **LSTM Layers**: Multiple LSTM layers for temporal modeling
  - Captures long-term dependencies in audio signals
  - Essential for modeling time-varying effects like distortion and compression
- **Output Layer**: Linear projection to output features
- **Activation**: Tanh activation to constrain output to [-1, 1] range (audio samples)

#### Parameters:
- `input_size`: Number of input features (default: 1 for mono audio)
- `hidden_size`: Number of hidden units (default: 96)
- `num_layers`: Number of LSTM layers (default: 2)
- `output_size`: Number of output features (default: 1 for mono audio)
- `dropout`: Dropout rate for regularization (default: 0.1)

#### Why LSTM?
- Guitar pedal effects often have memory (e.g., compression, tube saturation)
- LSTMs can model these temporal dependencies effectively
- Suitable for real-time processing with hidden state propagation

#### Alternative: WaveNetBlock
The module also includes a `WaveNetBlock` implementation for convolutional approaches:
- Better for capturing local temporal patterns
- Can be more efficient for certain types of effects
- Uses dilated convolutions with gated activations

### 2. Data Loading System

**File:** `neural_network/utils/data_loader.py`

#### AudioDataset Class:
Handles loading and preprocessing of paired audio data:
- Loads WAV files from input/target directories
- Segments audio into fixed-length sequences (default: 8192 samples)
- Converts stereo to mono if necessary
- Normalizes and prepares data for training

#### Key Features:
- **Automatic pairing**: Matches input and target files by name
- **Segmentation**: Splits long audio files into training sequences
- **Lazy loading**: Preloads and segments data for efficient access
- **PyTorch integration**: Returns tensors ready for model input

#### Data Preparation Workflow:
1. Record clean guitar input through audio interface
2. Record same guitar input processed through target pedal
3. Ensure both recordings are synchronized and same length
4. Place in appropriate train/val directories

### 3. Training System

**File:** `neural_network/utils/trainer.py`

#### Trainer Class:
Manages the complete training pipeline:

##### Features:
- **Loss Function**: Mean Squared Error (MSE) for audio regression
- **Optimizer**: Adam optimizer with configurable learning rate
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Gradient Clipping**: Prevents exploding gradients (max norm = 1.0)
- **Checkpointing**: Saves best model and periodic checkpoints
- **Training History**: Tracks losses and learning rates

##### Training Loop:
1. **Forward pass**: Process input through model
2. **Loss computation**: Calculate MSE between prediction and target
3. **Backward pass**: Compute gradients
4. **Gradient clipping**: Stabilize training
5. **Weight update**: Apply optimizer step
6. **Validation**: Evaluate on validation set
7. **Checkpointing**: Save best models

### 4. ONNX Export System

**File:** `neural_network/utils/onnx_export.py`

#### Purpose:
Converts trained PyTorch models to ONNX format for deployment in the CLAP plugin.

#### Key Functions:

##### `export_to_onnx()`
- Exports PyTorch model to ONNX format
- Supports dynamic batch size and sequence length
- Configurable opset version for compatibility

##### `verify_onnx_model()`
- Validates exported ONNX model
- Tests inference with ONNX Runtime
- Ensures output shapes match expectations

##### `optimize_onnx_model()`
- Applies ONNX optimizer passes
- Eliminates redundant operations
- Improves inference performance

##### `convert_checkpoint_to_onnx()`
- Convenience function to convert checkpoints
- Loads model weights and exports in one step

### 5. Configuration System

**File:** `neural_network/configs/default_config.py`

Centralized configuration for all training parameters:

#### MODEL_CONFIG
- Neural network architecture parameters
- Hidden size, number of layers, dropout rate

#### TRAINING_CONFIG
- Training hyperparameters
- Epochs, batch size, learning rate, sequence length

#### DATA_CONFIG
- Data paths and loading settings
- Train/validation directories, number of workers

#### EXPORT_CONFIG
- Model export settings
- Checkpoint and ONNX output directories

### 6. Training Script

**File:** `train.py`

Command-line interface for training models:

```bash
python train.py \
    --train-input data/raw/input/train \
    --train-target data/raw/target/train \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --export-onnx
```

#### Features:
- Flexible command-line arguments
- Automatic device selection (CPU/CUDA)
- Progress tracking and logging
- Automatic ONNX export after training

## Workflow

### Training Workflow

1. **Data Preparation**
   - Record clean guitar input
   - Record same input through target pedal
   - Place files in `data/raw/input/train` and `data/raw/target/train`
   - Optionally prepare validation set

2. **Model Training**
   ```bash
   python train.py --epochs 100 --export-onnx
   ```
   - Trains model on paired audio data
   - Saves checkpoints to `checkpoints/`
   - Exports best model to ONNX

3. **Model Evaluation**
   - Review training history in `checkpoints/training_history.json`
   - Listen to model predictions
   - Iterate on model architecture or training parameters

4. **Export for Plugin**
   - Best model automatically exported to `models/pedal_model.onnx`
   - Ready for integration with CLAP plugin

### Plugin Integration Workflow (Future)

1. **CLAP Plugin Development**
   - Implement CLAP plugin boilerplate
   - Integrate ONNX Runtime for inference
   - Load ONNX model at plugin initialization

2. **Real-time Inference**
   - Feed audio buffer to ONNX model
   - Process in chunks with hidden state management
   - Output processed audio to DAW

3. **Parameter Control**
   - Implement plugin parameters (gain, mix, etc.)
   - Map parameters to model inputs if needed
   - Provide UI for user interaction

## Technical Considerations

### Audio Processing
- **Sample Rate**: Default 44.1 kHz (configurable)
- **Bit Depth**: 32-bit float for training, plugin-dependent for deployment
- **Latency**: Depends on sequence length and model complexity
- **Buffer Size**: Should match plugin buffer for smooth real-time processing

### Model Performance
- **Parameters**: ~100K parameters for default config (96 hidden units, 2 layers)
- **Training Time**: Varies by dataset size (typically 1-2 hours on GPU)
- **Inference Speed**: LSTM can process ~100ms chunks in <1ms on modern CPUs

### Data Requirements
- **Minimum Data**: ~5-10 minutes of paired audio
- **Recommended**: 30-60 minutes for robust modeling
- **Diversity**: Include various playing styles, dynamics, and frequency content

### Best Practices

#### Data Recording
1. Use consistent input gain levels
2. Ensure perfect synchronization between input/target
3. Include silence regions for noise floor modeling
4. Vary playing dynamics (soft to hard picking)
5. Cover full frequency range (low E to high E string harmonics)

#### Training
1. Start with default hyperparameters
2. Monitor validation loss to prevent overfitting
3. Use gradient clipping to stabilize training
4. Save multiple checkpoints for comparison
5. A/B test models by listening to outputs

#### Model Selection
- **LSTM**: Good for most pedal types, especially time-varying effects
- **WaveNet**: Consider for simpler distortion/overdrive effects
- **Larger models**: More parameters = better quality but slower inference

## Future Enhancements

### Neural Network Side
- [ ] Multi-track training support
- [ ] Data augmentation techniques
- [ ] Alternative architectures (Transformers, TCN)
- [ ] Hyperparameter optimization
- [ ] Real-time training monitoring (TensorBoard)
- [ ] Model quantization for faster inference

### Plugin Side
- [ ] CLAP plugin implementation
- [ ] ONNX Runtime integration
- [ ] Real-time processing with <5ms latency
- [ ] Parameter mapping and automation
- [ ] Preset management system
- [ ] A/B comparison with reference pedal
- [ ] Multi-pedal chain modeling

## Dependencies

### Python Dependencies
See `requirements.txt` for complete list:
- **PyTorch**: Neural network framework
- **ONNX/ONNX Runtime**: Model export and inference
- **Soundfile**: Audio I/O
- **NumPy**: Numerical computing
- **tqdm**: Progress bars

### Plugin Dependencies (Future)
- **CLAP SDK**: Plugin framework
- **ONNX Runtime**: Model inference
- **JUCE** (optional): UI framework
- **Build system**: CMake, Make, or similar

## References

- [CLAP Framework Documentation](https://github.com/free-audio/clap)
- [ONNX Documentation](https://onnx.ai/onnx/)
- [PyTorch Audio Processing](https://pytorch.org/audio/stable/index.html)
- [WaveNet Paper](https://arxiv.org/abs/1609.03499)
- [Real-time Guitar Amplifier Emulation](https://arxiv.org/abs/2004.03282)

## License

[To be determined]

## Contributing

[To be determined]

## Contact

[To be determined]
