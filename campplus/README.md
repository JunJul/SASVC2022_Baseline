# CAM++ Speaker Verification

CAM++ (Class-Aware Multi-layer Aggregation with Pooling Plus) is a 7.2M-parameter speaker verification model producing 512-dimensional speaker embeddings. It replaces ECAPA-TDNN (21M params, 192-d) in this project.

Pretrained weights are from [3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker) (Apache 2.0), hosted on [ModelScope](https://www.modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k).

## Architecture

| Component | Details |
|---|---|
| Front-end (FCM) | 2D-CNN with ResBlocks, `m_channels=32`, input: 80-dim Fbank |
| TDNN | 128 channels, kernel=5, stride=2 |
| CAMDenseTDNN Blocks | 3 blocks (12/24/16 layers), growth_rate=32, dilation=1/2/2 |
| Pooling | Statistics pooling (mean + std) |
| Embedding | 512-d |

## File Structure

| File | Description |
|---|---|
| `DTDNN.py` | CAMPPlus model class (FCM + xvector backbone) |
| `layers.py` | Building blocks (TDNNLayer, CAMLayer, StatsPool, etc.) |
| `extractor.py` | Embedding extractor wrapper (waveform → fbank → embedding) |
| `CAMPPlusModel.py` | Training/evaluation wrapper (AMP, AAM-Softmax, EER eval) |
| `dataloader.py` | Custom dataloader supporting .m4a (ffmpeg) and .wav (soundfile) |
| `loss.py` | AAM-Softmax loss (margin=0.2, scale=30, 512-d embeddings) |

## Data Setup

All training data should be placed under `data/campplus/`:

```
data/campplus/
├── Voxceleb2_dev_wav/parta/    # Training data (.wav, 304k files, 1665 speakers)
├── vox1_test_wav/wav/          # VoxCeleb1-O test data (.wav)
├── musan_split/                # MUSAN augmentation (noise/speech/music)
├── RIRS_NOISES/simulated_rirs/ # Room impulse responses
├── train_list_campplus_wav.txt # Training list (speaker_id  relative/path.wav)
└── test_list.txt               # VoxCeleb1-O trial pairs
```

## Training

### 1. Generate Training List

```bash
python prepare_voxceleb2.py --src_dir data/campplus/Voxceleb2_dev_wav/parta --train_list data/campplus/train_list_campplus_wav.txt
```

### 2. Train (Fine-tune Pretrained)

```bash
python train_campplus_sv.py
```

Pretrained weights are auto-downloaded from ModelScope on first run.

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--max_epoch` | 15 | Number of epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--test_step` | 5 | Evaluate EER every N epochs |
| `--n_cpu` | 8 | Dataloader workers |
| `--grad_accum` | 1 | Gradient accumulation steps |

### 3. Resume Training

```bash
python train_campplus_sv.py --initial_model exps/campplus_sv/model/model_0010.model --max_epoch 30
```

### 4. Evaluate Only

```bash
python train_campplus_sv.py --eval --initial_model exps/campplus_sv/model/model_0010.model
```

## Results

Trained on VoxCeleb2 parta only (1665 speakers), evaluated on VoxCeleb1-O:

| Epoch | ACC | EER |
|-------|-----|-----|
| 5 | 76.65% | 1.77% |
| 10 | 80.16% | **1.62%** |
| 15 | 82.18% | 1.74% |

Best model: `exps/campplus_sv/model/model_0010.model` (EER 1.62%)

## Extracting Embeddings (for SASV)

```bash
python save_embeddings.py -campplus_weight exps/campplus_sv/model/model_0010.model
```

This extracts 512-d speaker embeddings for the SASV backend fusion pipeline.

## GPU Requirements

- **RTX 3050 4GB**: batch_size=32, AMP enabled
- **RTX 3060+ 8GB**: batch_size=64
- Training time: ~55 min/epoch with .wav files, ~15 epochs ≈ 14 hours
