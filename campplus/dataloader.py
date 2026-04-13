"""
DataLoader for CAM++ speaker verification training.
Reads .m4a files directly using ffmpeg (no pre-conversion needed).
Also supports .wav and .flac files via soundfile.
"""

import glob
import numpy
import os
import random
import struct
import subprocess
import soundfile
import torch
from pathlib import Path
from scipy import signal
from torch.utils.data import Dataset

# Locate ffmpeg binary
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_FFMPEG_CANDIDATES = [
    _SCRIPT_DIR / "tools" / "ffmpeg.exe",
    _SCRIPT_DIR / "tools" / "ffmpeg",
    "ffmpeg",
]
FFMPEG_BIN = "ffmpeg"
for _c in _FFMPEG_CANDIDATES:
    if Path(_c).is_file() if str(_c) != "ffmpeg" else True:
        FFMPEG_BIN = str(_c)
        break


def read_audio(filepath, sr=16000):
    """Read audio from any format. Uses ffmpeg for .m4a, soundfile for others."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".m4a", ".aac", ".mp4", ".mp3", ".opus", ".wma"):
        return _read_with_ffmpeg(filepath, sr)
    else:
        audio, file_sr = soundfile.read(filepath)
        if hasattr(audio, 'ndim') and audio.ndim > 1:
            audio = numpy.mean(audio, axis=1)
        return audio, file_sr


def _read_with_ffmpeg(filepath, sr=16000):
    """Decode audio file to 16kHz mono float32 numpy array via ffmpeg pipe."""
    cmd = [
        FFMPEG_BIN,
        "-i", str(filepath),
        "-f", "s16le",       # raw PCM signed 16-bit little-endian
        "-acodec", "pcm_s16le",
        "-ac", "1",          # mono
        "-ar", str(sr),      # target sample rate
        "-v", "error",       # suppress info output
        "pipe:1",            # pipe to stdout
    ]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        audio = numpy.frombuffer(proc.stdout, dtype=numpy.int16).astype(numpy.float32)
        audio = audio / 32768.0  # normalize to [-1, 1]
        return audio, sr
    except subprocess.CalledProcessError:
        raise RuntimeError(f"ffmpeg failed to read: {filepath}")


class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            normalized_path = file.replace('\\', '/')
            category = normalized_path.split('/')[-4]
            if category not in self.noiselist:
                self.noiselist[category] = []
            self.noiselist[category].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        try:
            audio, sr = read_audio(self.data_list[index])
        except Exception:
            return self.__getitem__((index + 1) % len(self.data_list))

        # Ensure mono audio
        if hasattr(audio, 'ndim') and audio.ndim > 1:
            audio = numpy.mean(audio, axis=1)
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        # Data Augmentation
        augtype = random.randint(0, 5)
        if augtype == 0:     # Original
            audio = audio
        elif augtype == 1:   # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:   # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:   # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:   # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:   # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


class EvalDataset(Dataset):
    """Evaluation dataset — uses soundfile for .wav files (VoxCeleb1 test is .wav)."""
    def __init__(self, file_list, eval_path, max_audio=300 * 160 + 240):
        self.file_list = file_list
        self.eval_path = eval_path
        self.max_audio = max_audio

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        audio, _ = read_audio(os.path.join(self.eval_path, file))

        # Full utterance
        data_1 = numpy.stack([audio], axis=0)

        # Splitted utterance matrix
        if audio.shape[0] <= self.max_audio:
            shortage = self.max_audio - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')

        feats = []
        startframe = numpy.linspace(0, audio.shape[0] - self.max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + self.max_audio])

        feats = numpy.stack(feats, axis=0).astype(float)

        return file, data_1, feats
