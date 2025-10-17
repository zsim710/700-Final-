#!/usr/bin/env python3
"""
Verify all files and paths are correctly set up for intelligibility-based averaging.
"""

import os
import json
from pathlib import Path

def check_file(path, description):
    """Check if a file exists."""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_directory(path, description):
    """Check if a directory exists."""
    exists = os.path.isdir(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def main():
    print("="*80)
    print("INTELLIGIBILITY-BASED AVERAGING - PRE-FLIGHT CHECK")
    print("="*80)
    
    all_good = True
    
    # Check scripts
    print("\n📜 Checking Scripts:")
    print("-"*80)
    scripts = [
        ("tools/average_sa_models_intelligibility.py", "Averaging script"),
        ("tools/run_intelligibility_averaging.py", "Orchestration script"),
        ("tools/evaluate_intelligibility_averaging.sh", "Evaluation script"),
        ("tools/summarize_intelligibility_results.py", "Summary script"),
        ("tools/quick_start_intelligibility_averaging.sh", "Quick start script"),
        ("tools/test_averaged_model_simple.py", "Inference script"),
    ]
    
    for script, desc in scripts:
        path = f"/home/zsim710/XDED/XDED/{script}"
        if not check_file(path, desc):
            all_good = False
    
    # Check speaker checkpoint mapping
    print("\n📋 Checking Speaker Checkpoint Mapping:")
    print("-"*80)
    checkpoint_json = "/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json"
    if check_file(checkpoint_json, "Speaker checkpoints JSON"):
        with open(checkpoint_json, 'r') as f:
            checkpoints = json.load(f)
        
        required_speakers_high = ["M09", "M14", "M10", "M08", "F05"]
        required_speakers_vl = ["M12", "M01", "F03", "M04"]
        
        print(f"\n  HIGH band speakers:")
        for speaker in required_speakers_high:
            if speaker in checkpoints:
                print(f"    ✅ {speaker}: {checkpoints[speaker]}")
            else:
                print(f"    ❌ {speaker}: MISSING")
                all_good = False
        
        print(f"\n  VERY_LOW band speakers:")
        for speaker in required_speakers_vl:
            if speaker in checkpoints:
                print(f"    ✅ {speaker}: {checkpoints[speaker]}")
            else:
                print(f"    ❌ {speaker}: MISSING")
                all_good = False
    else:
        all_good = False
    
    # Check test CSV paths
    print("\n📂 Checking Test CSV Files:")
    print("-"*80)
    test_csvs = {
        'M08': '/home/zsim710/partitions/uaspeech/by_speakers/M08.csv',
        'M01': '/home/zsim710/partitions/uaspeech/by_speakers/M01.csv',
    }
    
    for speaker, csv_path in test_csvs.items():
        if not check_file(csv_path, f"{speaker} test CSV"):
            all_good = False
    
    # Check base checkpoint files
    print("\n🏗️  Checking Base Checkpoint Files:")
    print("-"*80)
    base_ckpt = "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00"
    
    if check_directory(base_ckpt, "Base checkpoint directory (for normalizer)"):
        # Check required files with corrected paths
        required_files = {
            "hyperparams.yaml": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F02_E0D2/7775/hyperparams.yaml",
            "tokenizer.ckpt": "/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt",
            "normalizer.ckpt": os.path.join(base_ckpt, "normalizer.ckpt"),
        }
        
        for file_name, file_path in required_files.items():
            if not check_file(file_path, f"  {file_name}"):
                all_good = False
    else:
        all_good = False
    
    # Check output directories exist (will be created if not)
    print("\n📁 Output Directories (will be created if needed):")
    print("-"*80)
    output_dirs = [
        "/home/zsim710/XDED/XDED/results/intelligibility_averaging",
        "/home/zsim710/XDED/XDED/results/intelligibility_averaging/excluded",
        "/home/zsim710/XDED/XDED/results/intelligibility_averaging/included",
    ]
    
    for dir_path in output_dirs:
        exists = os.path.exists(dir_path)
        status = "✅ EXISTS" if exists else "📝 WILL BE CREATED"
        print(f"{status}: {dir_path}")
    
    # Check Python environment
    print("\n🐍 Checking Python Dependencies:")
    print("-"*80)
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch: NOT FOUND")
        all_good = False
    
    try:
        import speechbrain
        print(f"✅ SpeechBrain: Found")
    except ImportError:
        print("❌ SpeechBrain: NOT FOUND")
        all_good = False
    
    try:
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
    except ImportError:
        print("❌ Pandas: NOT FOUND")
        all_good = False
    
    try:
        import jiwer
        print(f"✅ jiwer: Found")
    except ImportError:
        print("❌ jiwer: NOT FOUND")
        all_good = False
    
    # Summary
    print("\n" + "="*80)
    if all_good:
        print("✅ PRE-FLIGHT CHECK PASSED")
        print("="*80)
        print("\nReady to run! Execute:")
        print("  cd /home/zsim710/XDED/XDED")
        print("  ./tools/quick_start_intelligibility_averaging.sh")
        print("\nOr run step-by-step:")
        print("  python3 tools/run_intelligibility_averaging.py")
        print("  ./tools/evaluate_intelligibility_averaging.sh")
        print("  python3 tools/summarize_intelligibility_results.py")
    else:
        print("❌ PRE-FLIGHT CHECK FAILED")
        print("="*80)
        print("\nPlease fix the issues marked with ❌ above.")
    print("")

if __name__ == "__main__":
    main()
