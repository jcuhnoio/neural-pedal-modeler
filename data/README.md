# Data Directory

This directory contains the training data for the neural pedal modeler.

## Directory Structure

```
data/
├── raw/              # Raw audio files
│   ├── input/        # Clean guitar input signals
│   │   ├── train/    # Training set inputs
│   │   └── val/      # Validation set inputs
│   └── target/       # Pedal-processed signals
│       ├── train/    # Training set targets
│       └── val/      # Validation set targets
└── processed/        # Preprocessed data (optional)
```

## Data Preparation

### Recording Setup

1. **Hardware Requirements:**
   - Audio interface with at least 2 inputs
   - Guitar
   - Target pedal to model
   - DI box (optional but recommended)

2. **Recording Process:**
   - Connect guitar to audio interface input 1
   - Connect guitar → pedal → audio interface input 2
   - Record both channels simultaneously
   - This ensures perfect synchronization

3. **Export to WAV:**
   - Export channel 1 (clean) to `input/train/recording_001.wav`
   - Export channel 2 (pedal) to `target/train/recording_001.wav`
   - Use matching filenames for paired recordings

### Audio Requirements

- **Format:** WAV (16-bit or 24-bit)
- **Sample Rate:** 44100 Hz (or consistent rate across all files)
- **Channels:** Mono (stereo will be converted automatically)
- **Length:** Any length (will be automatically segmented)

### Recommended Recording Content

For best results, record diverse guitar playing:

1. **Scales and arpeggios** - Full frequency range coverage
2. **Single notes** - Different dynamics (soft to hard picking)
3. **Chords** - Open and barre chords
4. **Sustained notes** - Test compression and sustain
5. **Palm muting** - Test response to dynamics
6. **Harmonics** - High-frequency content
7. **Silence** - Capture noise floor

### Data Split

- **Training:** 80% of recordings
- **Validation:** 20% of recordings

Place files in appropriate subdirectories:
- Training: `raw/input/train/` and `raw/target/train/`
- Validation: `raw/input/val/` and `raw/target/val/`

## Example File Naming

```
data/raw/input/train/
├── clean_001.wav
├── clean_002.wav
└── clean_003.wav

data/raw/target/train/
├── clean_001.wav  # Same filename as input
├── clean_002.wav
└── clean_003.wav
```

## Quick Start

If you don't have real recordings yet, you can create synthetic data for testing:

```python
import numpy as np
import soundfile as sf

# Create dummy audio
sample_rate = 44100
duration = 10  # seconds
samples = int(sample_rate * duration)

# Generate test signal (mix of frequencies)
t = np.linspace(0, duration, samples)
signal = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)

# Input (clean)
sf.write('data/raw/input/train/test_001.wav', signal, sample_rate)

# Target (with distortion)
distorted = np.tanh(signal * 3) * 0.5
sf.write('data/raw/target/train/test_001.wav', distorted, sample_rate)
```

## Tips

- Record at moderate input levels to avoid clipping
- Use high-quality audio interface for best results
- Keep pedal settings consistent across recordings
- Document pedal settings for reference
- Record room tone/noise floor separately if needed
