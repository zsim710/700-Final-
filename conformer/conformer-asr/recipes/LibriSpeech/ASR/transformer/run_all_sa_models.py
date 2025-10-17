#!/usr/bin/env python3
"""
Script to run each speaker-adaptive model and test it on all other speakers' datasets.
For each SA model, it tests on all speakers except itself.
"""

import os
import sys
import subprocess
import shutil
import yaml
from pathlib import Path

# Disable wandb to avoid interactive prompts
os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# All speakers
ALL_SPEAKERS = ["F02", "F03", "F04", "F05", "M01", "M04", "M05", "M07", "M08", "M09", "M10", "M11", "M12", "M14", "M16"]

# Checkpoint mapping for each speaker-adaptive model (excluding F02 since it's already done)
SA_MODEL_CHECKPOINTS = {
    #"F03": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F03_E0D3/7775/save/CKPT+2024-07-11+20-38-16+00",
    #"F04": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F04_E0D2/7775/save/CKPT+2024-07-11+17-39-01+00",
    #"F05": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F05_E0D2/7775/save/CKPT+2024-07-11+17-53-21+00",
    #"M01": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M01_E0D4/7775/save/CKPT+2024-07-12+18-22-00+00",
    #"M04": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M04_E0D2/7775/save/CKPT+2024-07-11+18-19-53+00",
    #"M05": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M05_E1D0/7775/save/CKPT+2024-07-12+11-58-04+00",
    #"M07": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M07_E0D0/7775/save/CKPT+2024-07-11+12-18-47+00",
    #"M08": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00",
    #"M09": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M09_E0D0/7775/save/CKPT+2024-07-11+12-42-43+00",
    "M10": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M10_E1D0/7775/save/CKPT+2024-07-12+12-51-27+00",
    "M11": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M11_E0D1/7775/save/CKPT+2024-07-11+16-21-04+00",
    "M12": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M12_E0D4/7775/save/CKPT+2024-07-12+20-01-22+00",
    "M14": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M14_E0D0/7775/save/CKPT+2024-07-11+13-35-20+00",
    "M16": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M16_E0D2/7775/save/CKPT+2024-07-11+20-10-30+00",
}

# Paths
YAML_FILE = "hparams/exp/uaspeech/ua_SA_val_uncommon_WRA.yaml"
BACKUP_FILE = YAML_FILE + ".backup"
TRAIN_SCRIPT = "train.py"

def backup_yaml():
    """Create a backup of the original YAML file."""
    shutil.copy2(YAML_FILE, BACKUP_FILE)
    print(f"Created backup: {BACKUP_FILE}")

def restore_yaml():
    """Restore the original YAML file from backup."""
    if os.path.exists(BACKUP_FILE):
        shutil.copy2(BACKUP_FILE, YAML_FILE)
        print(f"Restored original YAML file from backup")

def update_yaml_for_sa_model_and_test_speaker(sa_model, test_speaker):
    """Update the YAML file with the specified SA model and test speaker."""
    with open(YAML_FILE, 'r') as f:
        lines = f.readlines()
    
    # Update sa_model - find and replace the exact line
    for i, line in enumerate(lines):
        if line.strip().startswith('sa_model:'):
            # Preserve original indentation if any
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f'{indent}sa_model: {sa_model}\n'
            break
    
    # Update speaker
    for i, line in enumerate(lines):
        if line.strip().startswith('speaker:'):
            # Preserve original indentation if any
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f'{indent}speaker: {test_speaker}\n'
            break
    
    # Update checkpoint path
    checkpoint_path = SA_MODEL_CHECKPOINTS[sa_model]
    for i, line in enumerate(lines):
        if line.strip().startswith('load_ckpt:'):
            # Preserve original indentation if any
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f'{indent}load_ckpt: {checkpoint_path}\n'
            break
    
    # Update test_csv path - be very careful with list format
    for i, line in enumerate(lines):
        if '- !ref /home/zsim710/partitions/uaspeech/by_speakers/' in line and '.csv' in line:
            # Preserve exact indentation from the original line
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f'{indent}- !ref /home/zsim710/partitions/uaspeech/by_speakers/{test_speaker}.csv\n'
            break
    
    # Write the file back
    with open(YAML_FILE, 'w') as f:
        f.writelines(lines)
    
    print(f"Updated YAML file for SA model: {sa_model}, test speaker: {test_speaker}")

def run_test(sa_model, test_speaker):
    """Run the test for the specified SA model and test speaker."""
    print(f"\n{'='*70}")
    print(f"Testing SA model {sa_model} on speaker: {test_speaker}")
    print(f"{'='*70}")
    
    try:
        # Run the test command
        cmd = [sys.executable, TRAIN_SCRIPT, YAML_FILE, "--test_only"]
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Test completed successfully: SA model {sa_model} -> {test_speaker}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: SA model {sa_model} -> {test_speaker}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: SA model {sa_model} -> {test_speaker}")
        print(f"Error: {e}")
        return False

def get_test_speakers_for_sa_model(sa_model):
    """Get list of test speakers for a given SA model (all speakers except the SA model speaker)."""
    return [speaker for speaker in ALL_SPEAKERS if speaker != sa_model]

def main():
    """Main function to run tests for all SA models on all other speakers."""
    print("Starting Speaker-Adaptive model cross-testing automation...")
    
    # Check if required files exist
    if not os.path.exists(YAML_FILE):
        print(f"Error: YAML file not found: {YAML_FILE}")
        sys.exit(1)
    
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"Error: Training script not found: {TRAIN_SCRIPT}")
        sys.exit(1)
    
    # Check if checkpoints exist
    missing_checkpoints = []
    for sa_model, checkpoint_path in SA_MODEL_CHECKPOINTS.items():
        if not os.path.exists(checkpoint_path):
            missing_checkpoints.append(f"{sa_model}: {checkpoint_path}")
    
    if missing_checkpoints:
        print("Warning: The following checkpoints were not found:")
        for missing in missing_checkpoints:
            print(f"  {missing}")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Create backup
    backup_yaml()
    
    # Results tracking
    successful_tests = []
    failed_tests = []
    total_tests = 0
    
    try:
        # For each SA model
        for sa_model in SA_MODEL_CHECKPOINTS.keys():
            print(f"\n🚀 Starting tests for SA model: {sa_model}")
            
            # Get list of test speakers (all except the SA model speaker)
            test_speakers = get_test_speakers_for_sa_model(sa_model)
            
            print(f"Testing SA model {sa_model} on speakers: {', '.join(test_speakers)}")
            
            # Test on each speaker
            for test_speaker in test_speakers:
                total_tests += 1
                
                # Restore original YAML before updating
                restore_yaml()
                
                # Update YAML for current SA model and test speaker
                update_yaml_for_sa_model_and_test_speaker(sa_model, test_speaker)
                
                # Run test
                test_key = f"{sa_model}->{test_speaker}"
                if run_test(sa_model, test_speaker):
                    successful_tests.append(test_key)
                else:
                    failed_tests.append(test_key)
                
                print(f"Completed: {test_key}")
    
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    finally:
        # Always restore the original YAML file
        restore_yaml()
        
        # Clean up backup file
        if os.path.exists(BACKUP_FILE):
            os.remove(BACKUP_FILE)
            print("Cleaned up backup file")
    
    # Print summary
    print(f"\n{'='*80}")
    print("CROSS-TESTING SUMMARY")
    print(f"{'='*80}")
    print(f"Total SA models tested: {len(SA_MODEL_CHECKPOINTS)}")
    print(f"Total individual tests: {total_tests}")
    print(f"Successful tests: {len(successful_tests)}")
    print(f"Failed tests: {len(failed_tests)}")
    
    if successful_tests:
        print(f"\n✅ Successful tests:")
        for test in successful_tests:
            print(f"  {test}")
    
    if failed_tests:
        print(f"\n❌ Failed tests:")
        for test in failed_tests:
            print(f"  {test}")
    
    # Summary by SA model
    print(f"\n📊 SUMMARY BY SA MODEL:")
    for sa_model in SA_MODEL_CHECKPOINTS.keys():
        sa_successful = [t for t in successful_tests if t.startswith(f"{sa_model}->")]
        sa_failed = [t for t in failed_tests if t.startswith(f"{sa_model}->")]
        total_for_sa = len(sa_successful) + len(sa_failed)
        print(f"  {sa_model}: {len(sa_successful)}/{total_for_sa} successful")
    
    print(f"\nOriginal YAML file restored: {YAML_FILE}")

if __name__ == "__main__":
    main()
