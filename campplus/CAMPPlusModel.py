"""
CAM++ Speaker Verification Model — training & evaluation wrapper.
Mirrors ECAPAModel.py structure for compatibility with the existing training loop.
"""

import sys
import os
import time

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi
import tqdm
from torch.utils.data import DataLoader

from campplus.DTDNN import CAMPPlus
from campplus.loss import AAMsoftmax


class CAMPPlusModel(nn.Module):
    def __init__(self, lr, lr_decay, n_class, m, s, test_step,
                 feat_dim=80, embedding_size=512,
                 pretrained_path=None, **kwargs):
        super(CAMPPlusModel, self).__init__()

        self.feat_dim = feat_dim

        # Speaker encoder
        self.speaker_encoder = CAMPPlus(
            feat_dim=feat_dim,
            embedding_size=embedding_size,
        ).cuda()

        # Load pretrained weights if provided
        if pretrained_path is not None and os.path.isfile(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.speaker_encoder.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded pretrained CAM++ weights from {pretrained_path}")

        # Classifier
        self.speaker_loss = AAMsoftmax(
            n_class=n_class, embedding_size=embedding_size, m=m, s=s
        ).cuda()

        self.optim = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=test_step, gamma=lr_decay
        )
        self.scaler = torch.cuda.amp.GradScaler()

        n_params = sum(p.numel() for p in self.speaker_encoder.parameters()) / 1e6
        print(f"{time.strftime('%m-%d %H:%M:%S')} CAM++ params: {n_params:.2f}M")

    def _compute_fbank(self, waveform):
        """Compute 80-dim fbank from raw waveform batch on CPU (Kaldi-compatible).

        Args:
            waveform: (batch, samples) raw audio at 16kHz
        Returns:
            feats: (batch, T, feat_dim) on same device as waveform
        """
        device = waveform.device
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

        max_len = max(f.shape[0] for f in feats_list)
        feats = torch.zeros(len(feats_list), max_len, self.feat_dim)
        for i, feat in enumerate(feats_list):
            feats[i, : feat.shape[0], :] = feat

        # Mean normalization per-channel
        feats = feats - feats.mean(dim=1, keepdim=True)

        return feats.to(device)

    def train_network(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']

        for num, (data, labels) in enumerate(loader, start=1):
            self.optim.zero_grad()
            labels = torch.LongTensor(labels).cuda(non_blocking=True)
            data = data.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                feats = self._compute_fbank(data)
                speaker_embedding = self.speaker_encoder(feats)
                nloss, prec = self.speaker_loss(speaker_embedding, labels)

            if not torch.isfinite(nloss):
                print("Non-finite loss detected. Skipping batch.")
                continue

            self.scaler.scale(nloss).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            self.scaler.step(self.optim)
            self.scaler.update()

            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(
                time.strftime("%m-%d %H:%M:%S")
                + " [%2d] Lr: %5f, Training: %.2f%%,"
                % (epoch, lr, 100 * (num / loader.__len__()))
                + " Loss: %.5f, ACC: %2.2f%% \r"
                % (loss / num, top1 / index * len(labels))
            )
            sys.stderr.flush()

        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()

        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        # Reuse the CAM++ EvalDataset — supports .m4a and .wav
        from campplus.dataloader import EvalDataset

        eval_dataset = EvalDataset(setfiles, eval_path)
        eval_loader = DataLoader(
            eval_dataset, batch_size=1, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        with torch.no_grad():
            for file, data_1, data_2 in tqdm.tqdm(eval_loader, total=len(eval_loader)):
                file = file[0]
                data_1 = data_1.squeeze(0).float().cuda(non_blocking=True)
                data_2 = data_2.squeeze(0).float().cuda(non_blocking=True)

                with torch.cuda.amp.autocast():
                    # Full utterance
                    feats_1 = self._compute_fbank(data_1)
                    embedding_1 = self.speaker_encoder(feats_1)
                    embedding_1 = F.normalize(embedding_1, p=2, dim=1)

                    # 5 cropped utterances
                    feats_2 = self._compute_fbank(data_2)
                    embedding_2_list = self.speaker_encoder(feats_2)
                    embedding_2_list = F.normalize(embedding_2_list, p=2, dim=1)
                    embedding_2 = torch.mean(embedding_2_list, dim=0, keepdim=True)

                if not torch.isfinite(embedding_1).all() or not torch.isfinite(embedding_2).all():
                    print(f"Warning: NaN/Inf in embeddings for {file}. Using zeros.")
                    if not torch.isfinite(embedding_1).all():
                        embedding_1 = torch.zeros_like(embedding_1)
                    if not torch.isfinite(embedding_2).all():
                        embedding_2 = torch.zeros_like(embedding_2)

                embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []
        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Score = mean of 4 cosine similarities
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score_3 = torch.mean(torch.matmul(embedding_11, embedding_22.T))
            score_4 = torch.mean(torch.matmul(embedding_12, embedding_21.T))
            score = (score_1 + score_2 + score_3 + score_4) / 4
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        # EER + minDCF
        from ECAPATDNN.tools import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf

        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path, map_location="cpu")
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print(f"{origname} is not in the model.")
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print(f"Wrong parameter length: {origname}, model: {self_state[name].size()}, loaded: {loaded_state[origname].size()}")
                continue
            self_state[name].copy_(param)
