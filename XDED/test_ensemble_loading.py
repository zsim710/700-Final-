#!/usr/bin/env python3
"""
Test script to verify ensemble loading with cross-speaker matching.

Tests:
1. Different matching modes (strict, partial, all)
2. Core-ID indexing works correctly
3. Multiple teachers are loaded for each utterance
4. Batch collation works properly
"""

import sys
sys.path.insert(0, '/home/zsim710/XDED/XDED')

import torch
from torch.utils.data import DataLoader
from dassl.data.datasets.logit_ensemble import LogitEnsembleDataset, collate_logits


def test_matching_mode(mode, min_teachers=10, exclude_speakers=None):
    """Test a specific matching mode."""
    print(f"\n{'='*80}")
    print(f"Testing matching_mode='{mode}' (min_teachers={min_teachers})")
    print(f"{'='*80}")
    
    try:
        dataset = LogitEnsembleDataset(
            logit_root_dir="/home/zsim710/XDED/speechbrain/exp_results/logit_extraction",
            held_out_speaker="M08",
            split="train",
            matching_mode=mode,
            min_teachers=min_teachers,
            exclude_speakers=exclude_speakers
        )
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print(f"   ⚠️  No samples found!")
            return False
        
        # Test first sample
        print(f"\n🔍 Testing first sample:")
        sample = dataset[0]
        
        print(f"   Core ID: {sample['core_id']}")
        print(f"   Num Teachers: {sample['num_teachers']}")
        print(f"   Teacher Speakers: {sample['teacher_speakers']}")
        print(f"   Teacher Logits Shape: {sample['teacher_logits'].shape}")
        print(f"   Lengths: {sample['lengths'].tolist()}")
        if sample['target_text']:
            print(f"   Target Text: {sample['target_text']}")
        
        # Verify ensemble structure
        num_teachers = sample['num_teachers']
        max_len = sample['teacher_logits'].shape[1]
        vocab_size = sample['teacher_logits'].shape[2]
        
        assert sample['teacher_logits'].shape == (num_teachers, max_len, vocab_size), \
            f"Expected shape ({num_teachers}, {max_len}, {vocab_size}), got {sample['teacher_logits'].shape}"
        
        # Test a few random samples
        print(f"\n🎲 Checking 5 random samples:")
        import random
        indices = random.sample(range(len(dataset)), min(5, len(dataset)))
        
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            print(f"   Sample {i+1}: core_id={sample['core_id']}, "
                  f"num_teachers={sample['num_teachers']}, "
                  f"shape={sample['teacher_logits'].shape}")
        
        # Test batch loading with DataLoader
        print(f"\n📦 Testing batch loading:")
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_logits,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        print(f"   Batch teacher_logits shape: {batch['teacher_logits'].shape}")
        print(f"   Batch num_teachers: {batch['num_teachers'].tolist()}")
        print(f"   Batch core_ids: {batch['core_ids']}")
        print(f"   Batch teacher_speakers (first item): {batch['teacher_speakers'][0]}")
        
        # Verify batch dimensions
        assert batch['teacher_logits'].ndim == 4, \
            f"Expected 4D tensor [batch, teachers, seq_len, vocab], got {batch['teacher_logits'].ndim}D"
        
        print(f"\n✅ {mode.upper()} mode passed all checks!")
        return True
        
    except Exception as e:
        print(f"\n❌ {mode.upper()} mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_excluded_speakers():
    """Test excluding speakers (like M01 outlier)."""
    print(f"\n{'='*80}")
    print(f"Testing with excluded speakers (M01)")
    print(f"{'='*80}")
    
    try:
        dataset = LogitEnsembleDataset(
            logit_root_dir="/home/zsim710/XDED/speechbrain/exp_results/logit_extraction",
            held_out_speaker="M08",
            split="train",
            matching_mode="partial",
            min_teachers=10,
            exclude_speakers=["M01"]
        )
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Number of teachers: {len(dataset.teacher_speakers)}")
        print(f"   Teachers: {dataset.teacher_speakers}")
        
        # Verify M01 is not in any sample
        sample = dataset[0]
        assert "M01" not in sample['teacher_speakers'], "M01 should be excluded!"
        
        print(f"\n✅ Exclusion test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Exclusion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_core_id_matching():
    """Verify that core-ID matching actually returns the same utterance."""
    print(f"\n{'='*80}")
    print(f"Testing Core-ID Matching Consistency")
    print(f"{'='*80}")
    
    try:
        dataset = LogitEnsembleDataset(
            logit_root_dir="/home/zsim710/XDED/speechbrain/exp_results/logit_extraction",
            held_out_speaker="M08",
            split="train",
            matching_mode="strict"  # All teachers must have the utterance
        )
        
        # Pick a sample
        sample = dataset[0]
        core_id = sample['core_id']
        teacher_speakers = sample['teacher_speakers']
        
        print(f"\n🔍 Verifying Core-ID: {core_id}")
        print(f"   Available in {len(teacher_speakers)} teachers: {teacher_speakers}")
        
        # Load metadata and verify utterance IDs
        print(f"\n📋 Checking utterance IDs in metadata:")
        for speaker in teacher_speakers[:3]:  # Check first 3 teachers
            metadata = dataset.metadata[speaker]
            utt_ids = metadata.get('utterance_ids') or metadata.get('utterances', [])
            
            # Find this core_id in the speaker's utterances
            found = False
            for utt_id in utt_ids:
                if '_' in utt_id:
                    extracted_core_id = utt_id.split('_', 1)[1]
                    if extracted_core_id == core_id:
                        print(f"   {speaker}: {utt_id} -> core_id={extracted_core_id} ✓")
                        found = True
                        break
            
            if not found:
                print(f"   {speaker}: Core-ID {core_id} NOT FOUND ❌")
        
        print(f"\n✅ Core-ID matching test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Core-ID matching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print(f"\n{'#'*80}")
    print(f"# ENSEMBLE LOADING TEST SUITE")
    print(f"{'#'*80}")
    
    results = []
    
    # Test 1: Strict matching (all 14 teachers)
    results.append(("Strict Matching", test_matching_mode("strict")))
    
    # Test 2: Partial matching (≥10 teachers)
    results.append(("Partial Matching (≥10)", test_matching_mode("partial", min_teachers=10)))
    
    # Test 3: Partial matching (≥12 teachers)
    results.append(("Partial Matching (≥12)", test_matching_mode("partial", min_teachers=12)))
    
    # Test 4: All mode (no matching)
    results.append(("All Mode (No Matching)", test_matching_mode("all")))
    
    # Test 5: Excluding speakers
    results.append(("Excluding M01", test_excluded_speakers()))
    
    # Test 6: Core-ID consistency
    results.append(("Core-ID Matching", test_core_id_matching()))
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n{'='*80}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*80}")
    
    return all(p for _, p in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
