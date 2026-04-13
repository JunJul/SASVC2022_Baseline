import os
import pathlib

import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as Kaldi

from campplus.DTDNN import CAMPPlus


class CAMPPlusExtractor(nn.Module):
    """
    Wrapper for CAM++ speaker embedding extractor.
    Accepts raw waveform input (same interface as ECAPA-TDNN)
    and returns 512-d speaker embeddings.
    """

    def __init__(self, pretrained_path=None, feat_dim=80, embedding_size=512):
        super().__init__()
        self.model = CAMPPlus(feat_dim=feat_dim, embedding_size=embedding_size)
        self.feat_dim = feat_dim

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

    def _compute_fbank(self, waveform):
        """Compute 80-dim fbank features from raw waveform batch.

        Args:
            waveform: (batch, samples) raw audio at 16kHz
        Returns:
            feats: (batch, T, 80) fbank features with mean normalization
        """
        feats_list = []
        for i in range(waveform.shape[0]):
            wav = waveform[i : i + 1].cpu()
            feat = Kaldi.fbank(
                wav,
                num_mel_bins=self.feat_dim,
                sample_frequency=16000,
                dither=0,
            )
            feats_list.append(feat)

        # All utterances are padded to the same length (64600 samples),
        # so fbank features have the same number of frames.
        # Stack directly; pad as fallback for safety.
        max_len = max(f.shape[0] for f in feats_list)
        feats = torch.zeros(len(feats_list), max_len, self.feat_dim)
        for i, feat in enumerate(feats_list):
            feats[i, : feat.shape[0], :] = feat

        # Mean normalization (per-channel, over time)
        feats = feats - feats.mean(0, keepdim=True)

        return feats.to(waveform.device)

    def forward(self, x, aug=False):
        """
        Args:
            x: raw waveform tensor (batch, samples) at 16kHz
            aug: unused, kept for API compatibility with ECAPA-TDNN
        Returns:
            embeddings: (batch, 512)
        """
        feats = self._compute_fbank(x)
        embeddings = self.model(feats)
        return embeddings

    @staticmethod
    def download_pretrained(
        model_id="iic/speech_campplus_sv_en_voxceleb_16k",
        save_dir="./campplus/pretrained",
    ):
        """Download pretrained CAM++ weights from ModelScope.

        Returns the path to the downloaded weights file.
        """
        from modelscope.hub.snapshot_download import snapshot_download

        model_configs = {
            "iic/speech_campplus_sv_en_voxceleb_16k": {
                "revision": "v1.0.2",
                "model_pt": "campplus_voxceleb.bin",
            },
            "iic/speech_campplus_sv_zh-cn_16k-common": {
                "revision": "v1.0.0",
                "model_pt": "campplus_cn_common.bin",
            },
        }

        conf = model_configs[model_id]
        cache_dir = snapshot_download(model_id, revision=conf["revision"])
        weight_path = os.path.join(cache_dir, conf["model_pt"])
        print(f"[INFO] CAM++ weights downloaded to: {weight_path}")
        return weight_path
