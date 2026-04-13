#!/usr/bin/env python3
"""
Quick diagnostic script to verify training stability and identify NaN issues.
Run: python test_training.py
"""

import torch
import numpy as np
import sys
import os

# Add ECAPATDNN to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ECAPATDNN'))

from ECAPAModel import ECAPAModel
from dataLoader import train_loader
import argparse

def test_forward_pass():
    """Test forward pass and check for NaNs in embeddings"""
    print("=" * 60)
    print("Testing forward pass...")
    print("=" * 60)
    
    # Create a small model
    model = ECAPAModel(
        lr=5e-4,
        lr_decay=0.97,
        C=512,  # Smaller for testing
        n_class=100,  # Fewer classes for testing
        m=0.1,
        s=25,
        test_step=1
    )
    
    # Create dummy data
    batch_size = 4
    audio_length = 200 * 160 + 240  # Same as training
    
    dummy_audio = torch.randn(batch_size, audio_length).cuda()
    dummy_labels = torch.randint(0, 100, (batch_size,)).cuda()
    
    model.train()
    model.optim.zero_grad()
    
    # Forward pass
    with torch.amp.autocast(device_type='cuda'):
        embeddings = model.speaker_encoder.forward(dummy_audio, aug=False)
        print(f"✓ Embeddings shape: {embeddings.shape}")
        print(f"✓ Embedding values - min: {embeddings.min():.4f}, max: {embeddings.max():.4f}, mean: {embeddings.mean():.4f}")
        
        # Check for NaN/Inf
        if torch.isfinite(embeddings).all():
            print("✓ Embeddings are finite (no NaN/Inf)")
        else:
            print("✗ WARNING: Embeddings contain NaN/Inf!")
            return False
        
        # Compute loss
        loss, acc = model.speaker_loss.forward(embeddings, dummy_labels)
        print(f"✓ Loss: {loss.item():.6f}")
        print(f"✓ Accuracy: {acc.item():.4f}")
        
        if torch.isfinite(loss):
            print("✓ Loss is finite (no NaN/Inf)")
        else:
            print("✗ WARNING: Loss is NaN/Inf!")
            return False
    
    # Backward pass
    model.scaler.scale(loss).backward()
    model.scaler.unscale_(model.optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    model.scaler.step(model.optim)
    model.scaler.update()
    
    print("✓ Backward pass and optimization step completed successfully")
    return True

def test_eval_embeddings():
    """Test that evaluation embeddings don't have NaN/Inf"""
    print("\n" + "=" * 60)
    print("Testing evaluation embeddings...")
    print("=" * 60)
    
    model = ECAPAModel(
        lr=5e-4,
        lr_decay=0.97,
        C=512,
        n_class=100,
        m=0.1,
        s=25,
        test_step=1
    )
    model.eval()
    
    # Test full utterance
    dummy_full = torch.randn(1, 300 * 160 + 240).cuda()
    
    # Test 5 crops
    dummy_crops = torch.randn(5, 300 * 160 + 240).cuda()
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            emb_full = model.speaker_encoder.forward(dummy_full, aug=False)
            emb_full_norm = torch.nn.functional.normalize(emb_full, p=2, dim=1)
            
            emb_crops = model.speaker_encoder.forward(dummy_crops, aug=False)
            emb_crops_norm = torch.nn.functional.normalize(emb_crops, p=2, dim=1)
            
    print(f"✓ Full embedding shape: {emb_full_norm.shape}, finite: {torch.isfinite(emb_full_norm).all()}")
    print(f"✓ Crops embedding shape: {emb_crops_norm.shape}, finite: {torch.isfinite(emb_crops_norm).all()}")
    
    # Average crops
    emb_avg = torch.mean(emb_crops_norm, dim=0, keepdim=True)
    print(f"✓ Averaged embedding shape: {emb_avg.shape}, finite: {torch.isfinite(emb_avg).all()}")
    
    # Compute similarity score
    score = torch.sum(emb_full_norm * emb_avg, dim=1)
    print(f"✓ Similarity score: {score.item():.4f}, finite: {torch.isfinite(score)}")
    
    return torch.isfinite(emb_full_norm).all() and torch.isfinite(emb_crops_norm).all()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ECAPA-TDNN DIAGNOSTIC TEST")
    print("=" * 60)
    
    try:
        success1 = test_forward_pass()
        success2 = test_eval_embeddings()
        
        print("\n" + "=" * 60)
        if success1 and success2:
            print("✓ ALL TESTS PASSED - Ready to train!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("✗ SOME TESTS FAILED - Fix issues before training")
            print("=" * 60)
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
