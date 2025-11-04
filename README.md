# Neural Pedal Modeler

A neural network-based guitar pedal modeler that captures and replicates the sound characteristics of guitar effects pedals. The system trains deep learning models on paired audio data and exports them to ONNX format for real-time inference in a CLAP plugin.

## Overview

This project consists of two main components:

1. **Neural Network Training Pipeline** (Python/PyTorch) - Current implementation
2. **CLAP Plugin** (C++) - Future implementation

The neural network side trains LSTM-based models on recordings of clean guitar signals and their pedal-processed counterparts. Once trained, models are exported to ONNX format for deployment in the CLAP audio plugin.

## Features

- 🎸 **LSTM-based architecture** optimized for audio signal modeling
- 🔄 **Real-time capable** - processes audio in chunks with stateful inference
- 📦 **ONNX export** for cross-platform deployment
- 🎛️ **Flexible training** with configurable hyperparameters
- 📊 **Training monitoring** with loss tracking and checkpointing
- 🎵 **Audio-first design** with proper sample rate and normalization handling

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jcuhnoio/neural-pedal-modeler.git
cd neural-pedal-modeler

# Install dependencies
pip install -r requirements.txt
```

### Preparing Data

1. Record clean guitar input and pedal-processed output simultaneously
2. Export both channels as WAV files with matching names
3. Place files in the data directories:

```
data/raw/input/train/clean_001.wav
data/raw/target/train/clean_001.wav
```

See [data/README.md](data/README.md) for detailed recording instructions.

### Training a Model

```bash
# Train with default settings
python train.py --export-onnx

# Custom training
python train.py \
    --train-input data/raw/input/train \
    --train-target data/raw/target/train \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --export-onnx
```

### Running Inference

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input my_guitar.wav \
    --output processed_guitar.wav
```

## Project Structure

```
neural-pedal-modeler/
├── neural_network/          # Neural network training code
│   ├── models/              # Model architectures (PedalNet)
│   ├── utils/               # Training utilities and data loaders
│   └── configs/             # Configuration files
├── data/                    # Training data directory
│   └── raw/                 # Raw audio files (input/target)
├── models/                  # Exported ONNX models
├── checkpoints/             # Training checkpoints
├── train.py                 # Main training script
├── inference.py             # Inference script for testing
└── ARCHITECTURE.md          # Detailed architecture documentation
```

## Architecture

The system uses a recurrent neural network (LSTM) architecture designed for audio signal processing:

- **Input Layer**: Projects audio samples to hidden dimension
- **LSTM Layers**: Captures temporal dependencies and pedal memory effects
- **Output Layer**: Projects back to audio samples
- **Activation**: Tanh to constrain output to audio range [-1, 1]

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation of all components.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- ONNX Runtime
- NumPy
- Soundfile
- tqdm

See [requirements.txt](requirements.txt) for complete list.

## Training Tips

1. **Data Quality**: Use high-quality recordings with good signal-to-noise ratio
2. **Synchronization**: Ensure input and target recordings are perfectly aligned
3. **Diversity**: Record various playing styles and dynamics
4. **Duration**: At least 5-10 minutes of audio recommended
5. **Monitoring**: Check validation loss to prevent overfitting

## ONNX Export

Trained models are automatically exported to ONNX format:

```bash
python train.py --export-onnx --onnx-output models/my_pedal.onnx
```

The exported ONNX model includes:
- Dynamic batch size and sequence length
- Optimized for inference performance
- Compatible with ONNX Runtime
- Ready for CLAP plugin integration

## Future Work

- [ ] CLAP plugin implementation
- [ ] Real-time audio processing with <5ms latency
- [ ] Multiple pedal chain modeling
- [ ] Parameter control and automation
- [ ] Preset management system
- [ ] Alternative architectures (WaveNet, Transformers)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[To be determined]

## Acknowledgments

This project is inspired by research in neural audio processing and real-time guitar amplifier emulation.

## Contact

For questions or feedback, please open an issue on GitHub.
