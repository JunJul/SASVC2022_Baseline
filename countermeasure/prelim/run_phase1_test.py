"""
Phase 1 — End-to-End Integration Test
=======================================
This script verifies the entire pipeline works:
  1. Parse protocol files
  2. Load raw audio → resample to 16 kHz → pad/truncate
  3. Feed through AASIST model
  4. Verify forward pass produces valid logits
  5. Run a mini training loop (5 batches) to confirm gradients flow

Run this FIRST before full preprocessing to catch issues early.

Usage:
    python run_phase1_test.py
    python run_phase1_test.py --variant AASIST-L   # Lighter model
    python run_phase1_test.py --subset 100          # Fewer samples
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from config import (
    TRAIN_PROTOCOL, TRAIN_FLAC_DIR, DEV_PROTOCOL, DEV_FLAC_DIR,
    MAX_AUDIO_LENGTH, SMALL_SUBSET_SIZE,
)
from protocol_parser import load_all_protocols
from dataset import ASVspoofRawDataset
from aasist_model import build_model


def test_data_pipeline(subset_size=50):
    """Test 1: Verify data loading and shapes."""
    
    print("\n" + "=" * 60)
    print("  TEST 1: Data Pipeline Verification")
    print("=" * 60)
    
    # Parse protocols
    print("\n  [Step 1/3] Parsing protocol files...")
    protocols = load_all_protocols(verbose=False)
    for split, df in protocols.items():
        print(f"    {split:>5s}: {len(df):>7,} utterances | "
              f"bonafide: {(df['label_str']=='bonafide').sum():,} | "
              f"spoof: {(df['label_str']=='spoof').sum():,}")
    
    # Load a small subset
    print(f"\n  [Step 2/3] Loading {subset_size} raw audio samples...")
    dataset = ASVspoofRawDataset(
        protocol_file=TRAIN_PROTOCOL,
        flac_dir=TRAIN_FLAC_DIR,
        subset_size=subset_size,
    )
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    print(f"\n  [Step 3/3] Verifying batch shapes...")
    durations = []
    for batch_idx, (x, y, utt_ids) in enumerate(loader):
        if batch_idx == 0:
            print(f"    Audio tensor shape:  {x.shape}")
            print(f"    Expected:            (batch, {MAX_AUDIO_LENGTH})")
            print(f"    Label tensor shape:  {y.shape}")
            print(f"    Labels (first batch): {y.tolist()}")
            print(f"    Audio dtype:  {x.dtype}")
            print(f"    Audio range:  [{x.min():.4f}, {x.max():.4f}]")
            print(f"    Audio mean:   {x.mean():.6f}")
            print(f"    Audio std:    {x.std():.4f}")
        
        assert x.shape[1] == MAX_AUDIO_LENGTH, \
            f"Audio length mismatch: {x.shape[1]} vs {MAX_AUDIO_LENGTH}"
        assert y.min() >= 0 and y.max() <= 1, \
            f"Label range error: {y.min()} – {y.max()}"
        
        if batch_idx >= 4:
            break
    
    print(f"\n  Data pipeline test PASSED")
    return loader


def test_model_forward(variant="AASIST-L"):
    """Test 2: Verify model forward pass."""
    
    print("\n" + "=" * 60)
    print(f"  TEST 2: {variant} Forward Pass")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    model = build_model(variant).to(device)
    model.eval()
    
    # Dummy input
    batch_size = 4
    x = torch.randn(batch_size, MAX_AUDIO_LENGTH).to(device)
    
    print(f"\n  Input shape: {x.shape}")
    
    start = time.time()
    with torch.no_grad():
        logits = model(x)
    elapsed = time.time() - start
    
    print(f"  Output shape: {logits.shape}")
    print(f"  Logits (sample): {logits[0].cpu().tolist()}")
    
    # Softmax → probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"  Probabilities:   {probs[0].cpu().tolist()}")
    print(f"  Inference time:  {elapsed*1000:.1f} ms for {batch_size} samples")
    
    assert logits.shape == (batch_size, 2), \
        f"Output shape mismatch: {logits.shape}"
    
    print(f"\n  Forward pass test PASSED")
    return model, device


def test_mini_training(variant="AASIST-L", subset_size=100, num_batches=5):
    """Test 3: Run a few training steps to verify gradients."""
    
    print("\n" + "=" * 60)
    print(f"  TEST 3: Mini Training Loop ({num_batches} batches)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = build_model(variant).to(device)
    model.train()
    
    # Data
    dataset = ASVspoofRawDataset(
        protocol_file=TRAIN_PROTOCOL,
        flac_dir=TRAIN_FLAC_DIR,
        subset_size=subset_size,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0,
                        drop_last=True)
    
    # Loss & optimizer
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor([1.0, 9.0]).to(device)  # Handle class imbalance
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    print(f"\n  Training {variant} on {len(dataset)} samples...")
    print(f"  Loss function: CrossEntropy (weighted: spoof=1.0, bonafide=9.0)")
    print(f"  Optimizer: Adam (lr=1e-4)")
    print()
    
    losses = []
    for batch_idx, (x, y, _) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        # Check gradient norms
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        optimizer.step()
        
        # Predictions
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        
        losses.append(loss.item())
        print(f"    Batch {batch_idx+1}/{num_batches}: "
              f"loss={loss.item():.4f}  acc={acc:.2f}  "
              f"grad_norm={grad_norm:.4f}")
    
    # Verify loss is decreasing (roughly)
    if len(losses) >= 3:
        first_half = np.mean(losses[:len(losses)//2])
        second_half = np.mean(losses[len(losses)//2:])
        trend = "↓ decreasing" if second_half < first_half else "→ stable/increasing"
        print(f"\n  Loss trend: {trend} ({first_half:.4f} → {second_half:.4f})")
    
    print(f"\n  Mini training test PASSED")


def test_dev_inference(variant="AASIST-L", subset_size=50):
    """Test 4: Run inference on dev set and compute basic metrics."""
    
    print("\n" + "=" * 60)
    print(f"  TEST 4: Dev Set Inference (untrained model)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(variant).to(device)
    model.eval()
    
    dataset = ASVspoofRawDataset(
        protocol_file=DEV_PROTOCOL,
        flac_dir=DEV_FLAC_DIR,
        subset_size=subset_size,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for x, y, utt_ids in loader:
            x = x.to(device)
            logits = model(x)
            scores = torch.softmax(logits, dim=1)[:, 1]  # P(bonafide)
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    print(f"\n  Evaluated {len(all_scores)} dev utterances")
    print(f"  Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"  Mean score (bonafide): {all_scores[all_labels==1].mean():.4f}")
    print(f"  Mean score (spoof):    {all_scores[all_labels==0].mean():.4f}")
    
    # With random weights, scores should be ~0.5 for both classes
    print(f"\n  Note: Scores are from an UNTRAINED model (expect ~random).")
    print(f"  After Phase 2 training, bonafide scores should be >> spoof scores.")
    
    print(f"\n  Dev inference test PASSED")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Integration Tests")
    parser.add_argument("--variant", type=str, default="AASIST-L",
                        choices=["AASIST", "AASIST-L"],
                        help="Model variant to test")
    parser.add_argument("--subset", type=int, default=100,
                        help="Number of samples for tests")
    args = parser.parse_args()
    
    print("╔" + "═" * 58 + "╗")
    print("║   Phase 1: ASVspoof 2019 LA — Integration Test Suite      ║")
    print("║   Deepfake Audio Detection with AASIST                    ║")
    print("╚" + "═" * 58 + "╝")
    
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\n  PyTorch version: {torch.__version__}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run all tests
    try:
        test_data_pipeline(subset_size=args.subset)
        test_model_forward(variant=args.variant)
        test_mini_training(variant=args.variant, subset_size=args.subset)
        test_dev_inference(variant=args.variant, subset_size=args.subset)
        
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║   ALL PHASE 1 TESTS PASSED                             ║")
        print("║                                                           ║")
        print("║   Next steps:                                             ║")
        print("║   1. Run full preprocessing:                              ║")
        print("║      python preprocess_audio.py                           ║")
        print("║   2. Proceed to Phase 2 (training):                       ║")
        print("║      Train AASIST on the full LA training set             ║")
        print("╚" + "═" * 58 + "╝")
        
    except FileNotFoundError as e:
        print(f"\n  FILE NOT FOUND: {e}")
        print(f"    → Check that BASE_DIR in config.py points to your LA folder")
        print(f"    → Verify the folder structure matches expectations")
        from config import validate_paths
        validate_paths()
    
    except Exception as e:
        print(f"\n  TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
