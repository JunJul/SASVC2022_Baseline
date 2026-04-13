## Introduction
This repository implements a **Spoofing-Aware Speaker Verification (SASV)** system for the [SASV Challenge 2022](https://sasv-challenge.github.io). It combines:
- **CAM++** (7.2M params, 512-d) for speaker verification (replacing ECAPA-TDNN)
- **AASIST** for spoofing countermeasure
- **Baseline2** backend fusion DNN

The pipeline has 4 stages:
1. **Data preparation** — download datasets, convert audio, generate train lists
2. **Speaker verification training** — fine-tune CAM++ on VoxCeleb2
3. **Embedding extraction** — extract speaker + spoofing embeddings from ASVspoof2019
4. **SASV backend training** — train the fusion model and evaluate

### Prerequisites

#### Install requirements
```
pip install -r requirements.txt
```

#### Install ffmpeg (required for .m4a audio)
Place `ffmpeg.exe` in the `tools/` directory, or ensure it's available in your system PATH.

---

## Step 1: Data Preparation

### 1.1 Download ASVspoof2019 LA dataset
```
python ./aasist/download_dataset.py
```
This downloads the ASVspoof2019 LA dataset to `./LA/`.

### 1.2 Prepare VoxCeleb2 training data
VoxCeleb2 audio (`.m4a` format) should be placed under `data/campplus/Voxceleb2_dev_aac/` or pre-converted to `.wav`.

**Option A: Convert .m4a to .wav (recommended for faster training)**
```
python convert_to_wav.py --src data/campplus/Voxceleb2_dev_aac/parta --dst data/campplus/Voxceleb2_dev_wav/parta --workers 8
```
This generates `.wav` files and `data/campplus/train_list_campplus_wav.txt`.

**Option B: Use .m4a directly (no conversion, slower training)**
```
python prepare_voxceleb2.py --src_dir data/campplus/Voxceleb2_dev_aac/parta
```
The dataloader decodes `.m4a` on-the-fly via ffmpeg.

### 1.3 Data layout

```
data/campplus/
├── Voxceleb2_dev_wav/parta/    # Training audio (.wav, 304k files, 1665 speakers)
├── vox1_test_wav/wav/          # VoxCeleb1-O evaluation audio
├── musan_split/                # MUSAN augmentation (noise/speech/music)
├── RIRS_NOISES/simulated_rirs/ # Room impulse responses
├── train_list_campplus_wav.txt # Training list
└── test_list.txt               # VoxCeleb1-O trial pairs

protocols/                      # ASVspoof2019 protocols
├── ASVspoof2019.LA.cm.*.txt    # CM protocols (train/dev/eval)
└── ASVspoof2019.LA.asv.*.txt   # ASV protocols (dev/eval)

LA/                             # ASVspoof2019 audio (downloaded in Step 1.1)
├── ASVspoof2019_LA_train/
├── ASVspoof2019_LA_dev/
└── ASVspoof2019_LA_eval/
```

---

## Step 2: Train CAM++ Speaker Verification

Fine-tune a pretrained CAM++ model on VoxCeleb2. Pretrained weights are auto-downloaded from ModelScope on the first run.

```
python train_campplus_sv.py
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--max_epoch` | 15 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--test_step` | 5 | Evaluate EER every N epochs |
| `--n_cpu` | 8 | Number of dataloader workers |
| `--grad_accum` | 1 | Gradient accumulation steps |

**Resume training from a checkpoint:**
```
python train_campplus_sv.py --initial_model exps/campplus_sv/model/model_0010.model --max_epoch 30
```

**Evaluate only:**
```
python train_campplus_sv.py --eval --initial_model exps/campplus_sv/model/model_0010.model
```

Checkpoints and scores are saved to `exps/campplus_sv/`.

---

## Step 3: Extract Speaker & Spoofing Embeddings

Extract embeddings from the ASVspoof2019 dataset using the trained CAM++ and pretrained AASIST models. Embeddings are saved to `./embeddings/`.

```
python save_embeddings.py -campplus_weight exps/campplus_sv/model/model_0010.model
```

| Argument | Default | Description |
|---|---|---|
| `-aasist_config` | `./aasist/config/AASIST.conf` | AASIST config file |
| `-aasist_weight` | `./aasist/models/weights/AASIST.pth` | AASIST pretrained weights |
| `-campplus_weight` | auto-download | CAM++ weights (use your fine-tuned model) |

This produces:
- `embeddings/cm_embd_{trn,dev,eval}.pk` — 160-d spoofing embeddings (AASIST)
- `embeddings/asv_embd_{trn,dev,eval}.pk` — 512-d speaker embeddings (CAM++)
- `embeddings/spk_model_{dev,eval}.pk` — enrolled speaker models

**Using pre-extracted embeddings** (requires git-lfs):
```
git lfs install
git lfs pull
```

---

## Step 4: Train SASV Backend (Baseline2)

Train the fusion DNN that combines speaker and spoofing embeddings for joint SASV scoring.

```
python main.py --config ./configs/baseline2.conf
```

The Baseline2 model takes concatenated embeddings (512-d ASV + 512-d ASV + 160-d CM = 1184-d) and outputs a SASV score. Configuration is in `configs/baseline2.conf`.

Results (SASV-EER, SV-EER, SPF-EER) are logged during training.

---

## Project Structure

```
├── main.py                  # SASV backend training (Step 4)
├── save_embeddings.py       # Embedding extraction (Step 3)
├── train_campplus_sv.py     # CAM++ SV training (Step 2)
├── prepare_voxceleb2.py     # Generate train list from .m4a files
├── convert_to_wav.py        # Convert .m4a to .wav
├── metrics.py               # SASV-EER, SV-EER, SPF-EER calculation
├── utils.py                 # Utility functions
├── campplus/                # CAM++ speaker verification model
│   ├── DTDNN.py             # Model architecture (7.2M params)
│   ├── layers.py            # Building blocks
│   ├── extractor.py         # Embedding extractor wrapper
│   ├── CAMPPlusModel.py     # Training/eval wrapper
│   ├── dataloader.py        # Dataloader (.m4a/.wav support)
│   └── loss.py              # AAM-Softmax loss
├── aasist/                  # AASIST spoofing countermeasure
├── configs/                 # Configuration files
├── models/                  # Backend DNN architectures
├── systems/                 # PyTorch Lightning systems
├── protocols/               # ASVspoof2019 protocol files
├── embeddings/              # Extracted embeddings
└── data/campplus/           # Training/evaluation data
```

## Metrics

Use `get_all_EERs` in `metrics.py` to calculate all three EERs:
- **SASV-EER**: target vs (nontarget + spoof)
- **SV-EER**: target vs nontarget
- **SPF-EER**: target vs spoof

Protocols:
- Dev: `protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt`
- Eval: `protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt`

---

## References
[1] ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech
```bibtex
@article{wang2020asvspoof,
  title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
  author={Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and Delgado, H{\'e}ctor and Nautsch, Andreas and Evans, Nicholas and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and Lee, Kong Aik and others},
  journal={Computer Speech \& Language},
  volume={64},
  pages={101114},
  year={2020},
  publisher={Elsevier}
}
```
[2] AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
```bibtex
@inproceedings{Jung2022AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={Proc. ICASSP}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2022}
```
[3] ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification
```bibtex
@inproceedings{desplanques2020ecapa,
  title={{ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification}},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Proc. Interspeech 2020},
  pages={3830--3834},
  year={2020}
}
```
