# Deepfake Audio Detection with AASIST

A complete pipeline for detecting AI-generated (deepfake) speech using the **AASIST** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) architecture, trained and evaluated on the **ASVspoof 2019 Logical Access (LA)** dataset.

## Overview

This project implements an end-to-end system for distinguishing bonafide (real) speech from spoofed (synthesized/converted) speech. The pipeline covers data parsing, audio preprocessing, model training, and comprehensive evaluation with per-attack breakdowns.

**Key results on the ASVspoof 2019 LA eval set:**

| Metric | Value |
|---|---|
| Equal Error Rate (EER) | 10.35% |
| min t-DCF | 0.1269 |
| Accuracy @ EER threshold | 89.65% |
| Score separation | 0.7848 |
| Easiest attack (A13 — TTS/VC hybrid) | 0.14% EER |
| Hardest attack (A18 — TTS vocoder) | 24.18% EER |

## Project Structure

```
├── config.py               # Paths, hyperparameters, model configs
├── protocol_parser.py      # Parses ASVspoof CM protocol files
├── preprocess_audio.py     # Resamples & pads audio → .npy files
├── dataset.py              # PyTorch datasets (raw .flac or preprocessed .npy)
├── dataset_memory.py       # In-memory dataset variant (preloads all data to RAM)
├── aasist_model.py         # AASIST and AASIST-L model implementation
├── run_phase1_test.py      # Integration tests (data → model → gradient check)
├── train.py                # Training loop with EER eval, scheduling, checkpointing
├── evaluate.py             # Full evaluation with plots, per-attack analysis, DET curves
├── results_eval.json       # Evaluation results
└── scores_eval.txt         # Per-utterance scores for external tools
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU recommended (runs on CPU but slowly)

### Python Dependencies

```bash
pip install torch numpy librosa soundfile pandas scipy scikit-learn tqdm matplotlib
```

Optional:
```bash
pip install tensorboard   # for live training monitoring
```

## Dataset Setup

Download the **ASVspoof 2019 LA** dataset from the [ASVspoof website](https://www.asvspoof.org/) and organize it as follows:

```
LA/
├── ASVspoof2019_LA_train/flac/     # Training .flac files (LA_T_*.flac)
├── ASVspoof2019_LA_dev/flac/       # Development .flac files (LA_D_*.flac)
├── ASVspoof2019_LA_eval/flac/      # Evaluation .flac files (LA_E_*.flac)
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

Then update `BASE_DIR` in `config.py` to point to your `LA/` folder:

```python
BASE_DIR = Path("/path/to/your/LA")
```

## How to Run

### 1. Verify Setup (Phase 1)

Run integration tests to confirm everything works before committing to a full training run. This parses protocols, loads a small audio subset, runs a forward pass through the model, and verifies that gradients flow correctly.

```bash
python run_phase1_test.py
python run_phase1_test.py --variant AASIST-L --subset 100
```

### 2. Preprocess Audio (Optional but Recommended)

Converts raw `.flac` files to resampled, fixed-length `.npy` arrays for faster training. Audio is resampled to 16 kHz and padded/truncated to 64,600 samples (~4.04 seconds).

```bash
python preprocess_audio.py                    # Full preprocessing (all splits)
python preprocess_audio.py --split train      # Only train split
python preprocess_audio.py --subset 500       # Quick test with 500 samples
python preprocess_audio.py --workers 8        # More parallel workers
```

If you skip this step, training will load from raw `.flac` files on the fly (slower but requires no disk space for preprocessed data).

### 3. Train the Model (Phase 2)

```bash
# Default: train AASIST-L (lightweight variant)
python train.py

# Train full AASIST
python train.py --variant AASIST

# Use preprocessed .npy files (faster I/O)
python train.py --use_preprocessed

# Custom hyperparameters
python train.py --epochs 50 --batch_size 24 --lr 1e-4

# Quick experiment on a small subset
python train.py --subset 5000

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pt

# Enable mixed precision (GPU only)
python train.py --use_amp
```

**Training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--variant` | `AASIST-L` | Model variant (`AASIST` or `AASIST-L`) |
| `--epochs` | `100` | Maximum training epochs |
| `--batch_size` | `24` | Batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--weight_decay` | `1e-4` | L2 regularization |
| `--warmup_epochs` | `5` | Linear LR warmup epochs |
| `--spoof_weight` | `1.0` | Cross-entropy weight for spoof class |
| `--bonafide_weight` | `9.0` | Cross-entropy weight for bonafide class |
| `--patience` | `20` | Early stopping patience |
| `--use_preprocessed` | `False` | Load from `.npy` instead of `.flac` |
| `--use_amp` | `False` | Mixed precision training |
| `--subset` | `None` | Limit samples per split |

Training saves the best model (by dev EER) to `checkpoints/best_model.pt`, generates loss/EER curves, and optionally logs to TensorBoard.

### 4. Evaluate (Phase 3)

```bash
# Evaluate on the eval set
python evaluate.py --checkpoint checkpoints/best_model.pt

# Evaluate on dev set
python evaluate.py --checkpoint checkpoints/best_model.pt --split dev

# Generate all plots
python evaluate.py --checkpoint checkpoints/best_model.pt --plot
```

Evaluation produces:
- Overall EER and min t-DCF
- Per-attack EER breakdown (A07–A19)
- DET curve
- Score distribution histograms
- A multi-panel summary figure
- A scores file compatible with official ASVspoof evaluation tools

## Model Architecture

AASIST processes raw audio waveforms through a **Sinc-based convolutional front-end** (learned bandpass filters), followed by **dual-branch encoding** (spectral and temporal paths using residual blocks), and integrates the two branches via **Graph Attention Networks** for final classification.

Two variants are available:

| Variant | Parameters | Use Case |
|---|---|---|
| `AASIST` | ~300K+ | Full model, best accuracy |
| `AASIST-L` | ~30K+ | Lightweight, faster training, good for prototyping |

## Audio Preprocessing Details

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Max length | 64,600 samples (~4.04 s) |
| Padding mode | Wrap (tiles audio from the beginning) |
| Input format | Raw waveform (no handcrafted features) |

## Class Imbalance Handling

The ASVspoof 2019 LA dataset is heavily imbalanced (far more spoof than bonafide samples). This is addressed with **weighted cross-entropy loss** — the default configuration weights bonafide samples 9× higher than spoof samples to prevent the model from trivially predicting spoof for everything.

## Results

The trained AASIST-L model achieves an overall **EER of 10.35%** on the eval set. Performance varies significantly across attack types: traditional vocoder-based and hybrid attacks (A09, A11, A13, A14) are detected with near-perfect accuracy (EER < 2.5%), while certain voice conversion (A17) and advanced vocoder attacks (A18) remain challenging at 21–24% EER.

## Reference

> Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," *ICASSP 2022*.
> GitHub: https://github.com/clovaai/aasist

## License

This implementation is for academic and research use. The ASVspoof 2019 dataset has its own usage terms — see the [ASVspoof website](https://www.asvspoof.org/) for details.
