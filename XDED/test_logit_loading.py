#!/usr/bin/env python3
"""
Test script for LogitEnsembleDataset - runs without importing dassl to avoid circular imports
"""

import os
import sys
import json
import torch

# Remove XDED from path to avoid ssl circular import
sys.path = [p for p in sys.path if 'XDED' not in p]

logit_root_dir = "/home/zsim710/XDED/speechbrain/exp_results/logit_extraction"

print("🧪 Testing Logit Loading...")
print("=" * 70)

# Define intelligibility levels (CORRECTED)
intelligibility_map = {
    "VERY_LOW": ["M04", "F03", "M12", "M01"],
    "LOW": ["M07", "F02", "M16"],
    "MID": ["M05", "M11", "F04"],
    "HIGH": ["M09", "M14", "M10", "M08", "F05"]
}

# Test configuration - Cross-intelligibility ensemble (ALL speakers except held-out)
all_speakers = ["F02", "F03", "F04", "F05", "M01", "M04", "M05", 
                "M07", "M08", "M09", "M10", "M11", "M12", "M14", "M16"]

test_config = {
    "M08_FOLD": {
        "held_out": "M08",
        "teachers": [spk for spk in all_speakers if spk != "M08"],  # All 14 others
        "description": "Cross-intelligibility ensemble: All 14 speakers → M08"
    }
}

def get_logit_path(speaker, logit_type="decoder"):
    """Get path to logit file."""
    return os.path.join(
        logit_root_dir,
        speaker,
        speaker,
        f"{speaker}_{logit_type}_logits.pt"
    )

def get_metadata_path(speaker):
    """Get path to metadata file."""
    return os.path.join(
        logit_root_dir,
        speaker,
        speaker,
        f"{speaker}_metadata.json"
    )

print("\n📊 Intelligibility Level Configuration (CORRECTED):")
print("-" * 70)
for level, speakers in intelligibility_map.items():
    print(f"{level}: {speakers}")

print("\n\n🎯 Fold Configuration:")
print("-" * 70)
for fold_name, config in test_config.items():
    print(f"\n{fold_name}:")
    print(f"  {config['description']}")
    print(f"  Held-out (test): {config['held_out']}")
    print(f"  Teachers (train): {len(config['teachers'])} speakers")
    print(f"  Teachers: {config['teachers']}")

print("\n\n🔍 Testing Logit File Accessibility:")
print("-" * 70)

total_utterances = {}

for fold_name, config in test_config.items():
    print(f"\n{fold_name}:")
    
    # Test held-out speaker
    held_out = config['held_out']
    logit_path = get_logit_path(held_out)
    metadata_path = get_metadata_path(held_out)
    
    if os.path.exists(logit_path) and os.path.exists(metadata_path):
        try:
            logits = torch.load(logit_path, map_location='cpu', weights_only=False)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"  ✅ Held-out {held_out}:")
            print(f"     - Logits: {len(logits)} utterances")
            if len(logits) > 0:
                print(f"     - Logit shape: {logits[0].shape}")
            print(f"     - Path: {logit_path}")
            
            total_utterances[held_out] = len(logits)
        except Exception as e:
            print(f"  ❌ Error loading {held_out}: {e}")
    else:
        print(f"  ❌ Files not found for {held_out}")
    
    # Test teacher speakers
    print(f"\n  Teacher speakers ({len(config['teachers'])}):")
    for teacher in config['teachers']:
        logit_path = get_logit_path(teacher)
        metadata_path = get_metadata_path(teacher)
        
        if os.path.exists(logit_path) and os.path.exists(metadata_path):
            try:
                logits = torch.load(logit_path, map_location='cpu', weights_only=False)
                print(f"    ✅ {teacher}: {len(logits)} utterances")
                total_utterances[teacher] = len(logits)
            except Exception as e:
                print(f"    ❌ Error loading {teacher}: {e}")
        else:
            print(f"    ❌ Files not found for {teacher}")

print("\n\n📈 Summary Statistics:")
print("-" * 70)
for fold_name, config in test_config.items():
    print(f"\n{fold_name}:")
    
    # Calculate total training samples
    train_samples = sum(total_utterances.get(spk, 0) for spk in config['teachers'])
    test_samples = total_utterances.get(config['held_out'], 0)
    
    print(f"  Training samples (14 teachers): {train_samples}")
    print(f"  Test samples ({config['held_out']}): {test_samples}")
    print(f"  Train/Test ratio: {train_samples / test_samples if test_samples > 0 else 0:.2f}x")

print("\n\n🎯 Next Steps:")
print("-" * 70)
print("""
1. Load all 14 teacher logits for each utterance
2. Create ensemble target (simple average or learned weights)
3. Train student model to match ensemble using KL divergence
4. Evaluate student on M08 held-out test set
5. Repeat for other folds: M05, M16, M01
""")

print("\n✅ Logit loading test complete!")
