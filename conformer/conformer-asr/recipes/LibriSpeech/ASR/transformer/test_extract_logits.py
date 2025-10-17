#!/usr/bin/env python3
"""
Test script to extract logits from a single SA model to verify the implementation.
"""

import os
import sys

# Set environment variables
os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def test_single_model():
    """Test extraction on a single SA model."""
    
    # Import the extraction script
    from extract_sa_logits import extract_logits_for_sa_model
    
    # Test parameters
    test_model = "F03"  # Start with F03
    hparams_file = "hparams/exp/uaspeech/ua_SA_val_uncommon_WRA.yaml"
    output_dir = "/tmp/test_logit_extraction"
    
    print(f"Testing decoder logit extraction for SA model: {test_model}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract logits for the test model
    result = extract_logits_for_sa_model(test_model, hparams_file, output_dir)
    
    if result:
        print(f"✅ Test successful! Logits extracted for {test_model}")
        
        # Check if files were created
        expected_file = f"{output_dir}/{test_model}/{test_model}/{test_model}_decoder_logits.pt"
        metadata_file = f"{output_dir}/{test_model}/{test_model}/{test_model}_metadata.json"
        
        if os.path.exists(expected_file):
            file_size = os.path.getsize(expected_file) / (1024*1024)  # MB
            print(f"📁 Decoder logits file: {expected_file} ({file_size:.1f} MB)")
        else:
            print(f"❌ Expected file not found: {expected_file}")
            
        if os.path.exists(metadata_file):
            print(f"📄 Metadata file: {metadata_file}")
        else:
            print(f"❌ Metadata file not found: {metadata_file}")
    else:
        print(f"❌ Test failed for {test_model}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing SA model decoder logit extraction...")
    
    success = test_single_model()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("You can now run the full extraction with:")
        print("python extract_sa_logits.py")
    else:
        print("\n💥 Test failed. Please check the logs for errors.")
        sys.exit(1)