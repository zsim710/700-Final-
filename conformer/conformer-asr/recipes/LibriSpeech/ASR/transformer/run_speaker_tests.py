#!/usr/bin/env python3
"""
Script to run tests for multiple speakers by updating the YAML config file
and executing the test for each speaker.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# List of speakers to test
SPEAKERS = ["F03", "F04", "F05", "M01", "M04", "M05", "M07", "M08", "M09", "M10", "M12", "M14", "M16"]

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

def update_yaml_for_speaker(speaker):
    """Update the YAML file with the specified speaker."""
    with open(YAML_FILE, 'r') as f:
        content = f.read()
    
    # Replace speaker placeholder
    content = content.replace('speaker: <speaker>', f'speaker: {speaker}')
    
    # Replace test_csv path
    old_test_csv = """test_csv:
    - !ref /home/zsim710/partitions/uaspeech/by_speakers/<speaker>.csv"""
    new_test_csv = f"""test_csv:
    - !ref /home/zsim710/partitions/uaspeech/by_speakers/{speaker}.csv"""
    
    content = content.replace(old_test_csv, new_test_csv)
    
    with open(YAML_FILE, 'w') as f:
        f.write(content)
    
    print(f"Updated YAML file for speaker: {speaker}")

def run_test(speaker):
    """Run the test for the specified speaker."""
    print(f"\n{'='*50}")
    print(f"Running test for speaker: {speaker}")
    print(f"{'='*50}")
    
    try:
        # Run the test command
        cmd = [sys.executable, TRAIN_SCRIPT, YAML_FILE, "--test_only"]
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Test completed successfully for speaker: {speaker}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed for speaker: {speaker}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error for speaker: {speaker}")
        print(f"Error: {e}")
        return False

def main():
    """Main function to run tests for all speakers."""
    print("Starting speaker test automation...")
    
    # Check if required files exist
    if not os.path.exists(YAML_FILE):
        print(f"Error: YAML file not found: {YAML_FILE}")
        sys.exit(1)
    
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"Error: Training script not found: {TRAIN_SCRIPT}")
        sys.exit(1)
    
    # Create backup
    backup_yaml()
    
    # Results tracking
    successful_tests = []
    failed_tests = []
    
    try:
        for speaker in SPEAKERS:
            # Restore original YAML before updating for next speaker
            restore_yaml()
            
            # Update YAML for current speaker
            update_yaml_for_speaker(speaker)
            
            # Run test
            if run_test(speaker):
                successful_tests.append(speaker)
            else:
                failed_tests.append(speaker)
            
            print(f"Completed speaker: {speaker}")
    
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
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total speakers tested: {len(SPEAKERS)}")
    print(f"Successful tests: {len(successful_tests)}")
    print(f"Failed tests: {len(failed_tests)}")
    
    if successful_tests:
        print(f"\n✅ Successful speakers: {', '.join(successful_tests)}")
    
    if failed_tests:
        print(f"\n❌ Failed speakers: {', '.join(failed_tests)}")
    
    print(f"\nOriginal YAML file restored: {YAML_FILE}")

if __name__ == "__main__":
    main()
