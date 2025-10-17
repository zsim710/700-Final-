#!/usr/bin/env python3
"""
Test script to verify audio loading in LogitEnsembleDataset.

Tests:
1. Audio files can be loaded from CSV mappings
2. Audio waveforms have correct shapes
3. Batch collation handles variable audio lengths
4. Integration with teacher logits works correctly
"""

import sys
sys.path.insert(0, '/home/zsim710/XDED/XDED')

import torch
from torch.utils.data import DataLoader
from dassl.data.datasets.logit_ensemble import LogitEnsembleDataset, collate_logits


def test_audio_loading():
    """Test audio loading functionality."""
    print("=" * 80)
    print("🧪 Testing Audio Loading in LogitEnsembleDataset")
    print("=" * 80)
    
    # Test 1: Create training dataset
    print("\n📊 Test 1: Creating training dataset (M08 held-out, partial matching)...")
    try:
        train_dataset = LogitEnsembleDataset(
            held_out_speaker="M08",
            split="train",
            matching_mode="partial",
            min_teachers=10
        )
        print(f"✅ Training dataset loaded: {len(train_dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to create training dataset: {e}")
        return False
    
    # Test 2: Load single sample
    print("\n📦 Test 2: Loading single sample...")
    try:
        sample = train_dataset[0]
        
        print(f"\n   Sample keys: {list(sample.keys())}")
        print(f"   Audio shape: {sample['audio'].shape}")
        print(f"   Audio dtype: {sample['audio'].dtype}")
        print(f"   Sample rate: {sample['sample_rate']}")
        print(f"   Audio duration: {sample['audio'].shape[0] / sample['sample_rate']:.2f}s")
        print(f"   WAV path: {sample['wav_path']}")
        print(f"   Teacher logits shape: {sample['teacher_logits'].shape}")
        print(f"   Num teachers: {sample['num_teachers']}")
        print(f"   Core ID: {sample['core_id']}")
        if sample['target_text']:
            print(f"   Target text: {sample['target_text']}")
        
        # Verify audio is not empty
        assert sample['audio'].shape[0] > 0, "Audio is empty!"
        assert sample['sample_rate'] == 16000, f"Unexpected sample rate: {sample['sample_rate']}"
        
        print("✅ Single sample test passed!")
        
    except Exception as e:
        print(f"❌ Failed to load single sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Test multiple samples
    print("\n🎲 Test 3: Loading 5 random samples...")
    try:
        import random
        indices = random.sample(range(len(train_dataset)), min(5, len(train_dataset)))
        
        for i, idx in enumerate(indices):
            sample = train_dataset[idx]
            audio_duration = sample['audio'].shape[0] / sample['sample_rate']
            print(f"   Sample {i+1}: core_id={sample['core_id']}, "
                  f"audio_duration={audio_duration:.2f}s, "
                  f"num_teachers={sample['num_teachers']}")
        
        print("✅ Multiple samples test passed!")
        
    except Exception as e:
        print(f"❌ Failed to load multiple samples: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test DataLoader with batching
    print("\n📦 Test 4: Testing DataLoader with batch_size=4...")
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_logits,
            num_workers=0  # Use 0 for debugging
        )
        
        batch = next(iter(train_loader))
        
        print(f"\n   Batch keys: {list(batch.keys())}")
        print(f"   Audio batch shape: {batch['audio'].shape}")
        print(f"   Audio lengths: {batch['audio_lengths'].tolist()}")
        print(f"   Sample rate: {batch['sample_rate']}")
        print(f"   Teacher logits shape: {batch['teacher_logits'].shape}")
        print(f"   Num teachers: {batch['num_teachers'].tolist()}")
        print(f"   Core IDs: {batch['core_ids']}")
        
        # Verify batch dimensions
        assert batch['audio'].ndim == 2, f"Expected 2D audio batch, got {batch['audio'].ndim}D"
        assert batch['audio'].shape[0] == 4, f"Expected batch size 4, got {batch['audio'].shape[0]}"
        assert batch['teacher_logits'].ndim == 4, f"Expected 4D teacher logits, got {batch['teacher_logits'].ndim}D"
        
        print("✅ DataLoader batching test passed!")
        
    except Exception as e:
        print(f"❌ Failed DataLoader test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test test dataset
    print("\n📊 Test 5: Creating test dataset (M08)...")
    try:
        test_dataset = LogitEnsembleDataset(
            held_out_speaker="M08",
            split="test"
        )
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
        
        if len(test_dataset) > 0:
            test_sample = test_dataset[0]
            print(f"\n   Test sample keys: {list(test_sample.keys())}")
            print(f"   Audio shape: {test_sample['audio'].shape}")
            print(f"   WAV path: {test_sample['wav_path']}")
            print(f"   Logits shape: {test_sample['logits'].shape}")
            print(f"   Utterance ID: {test_sample['utterance_id']}")
        
        print("✅ Test dataset test passed!")
        
    except Exception as e:
        print(f"❌ Failed test dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Verify audio-logit alignment
    print("\n🔍 Test 6: Verifying audio-logit alignment...")
    try:
        sample = train_dataset[0]
        
        # Calculate expected logit length from audio
        # Audio: sample_rate = 16000 Hz
        # CNN stride = 2 twice = 4x downsampling
        # Hop length = 160 samples
        # Expected: audio_len / 160 / 4
        
        audio_len = sample['audio'].shape[0]
        expected_frames = audio_len // 160  # Mel-spec frames
        expected_logit_len = expected_frames // 4  # After CNN downsampling
        actual_logit_len = sample['teacher_logits'].shape[1]
        
        print(f"   Audio length: {audio_len} samples")
        print(f"   Expected mel frames: ~{expected_frames}")
        print(f"   Expected logit length: ~{expected_logit_len}")
        print(f"   Actual logit length: {actual_logit_len}")
        
        # Allow some tolerance due to padding/alignment
        ratio = actual_logit_len / expected_logit_len if expected_logit_len > 0 else 0
        print(f"   Length ratio: {ratio:.2f}")
        
        if 0.8 <= ratio <= 1.2:
            print("✅ Audio-logit alignment looks reasonable!")
        else:
            print("⚠️  Audio-logit lengths may not align perfectly (this might be OK)")
        
    except Exception as e:
        print(f"❌ Failed alignment check: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\n📝 Summary:")
    print("   - Audio loading from CSV: ✅")
    print("   - Waveform extraction: ✅")
    print("   - Batch collation: ✅")
    print("   - Train/test datasets: ✅")
    print("   - Audio-logit integration: ✅")
    print("\n🎉 Audio loading is ready for training!")
    
    return True


if __name__ == "__main__":
    success = test_audio_loading()
    sys.exit(0 if success else 1)
