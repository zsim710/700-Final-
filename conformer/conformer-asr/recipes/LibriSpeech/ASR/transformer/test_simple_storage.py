#!/usr/bin/env python3
"""Simple test to check logit storage mechanism"""

import torch
import json
import os

class TestLogitStorage:
    def __init__(self):
        self.logit_storage = {
            'ctc_logits': [],           
            'decoder_logits': [],       
            'wav_lens': [],             
            'token_lens': [],           
            'utterance_ids': [],        
            'target_tokens': [],        
            'target_words': [],         
        }

    def save_extracted_logits(self, output_dir, model_name):
        """Save the extracted logits to files"""
        print(f"Attempting to save logits for {model_name}")
        
        # First check what we have in storage
        print(f"Storage keys: {list(self.logit_storage.keys())}")
        for key, value in self.logit_storage.items():
            print(f"{key}: type={type(value)}, length={len(value) if hasattr(value, '__len__') else 'N/A'}")
            if hasattr(value, '__len__') and len(value) > 0:
                print(f"  First item type: {type(value[0])}")
                if hasattr(value[0], 'shape'):
                    print(f"  First item shape: {value[0].shape}")
        
        try:
            # Check if we have any logits
            if not self.logit_storage['ctc_logits'] and not self.logit_storage['decoder_logits']:
                print(f"No logits found in storage for {model_name}")
                return False
            
            # Concatenate logits and save
            if self.logit_storage['ctc_logits']:
                try:
                    print(f"Concatenating {len(self.logit_storage['ctc_logits'])} CTC logits")
                    ctc_logits = torch.cat(self.logit_storage['ctc_logits'], dim=0)
                    print(f"CTC logits concatenated: {ctc_logits.shape}")
                    torch.save(ctc_logits, f"{output_dir}/{model_name}_ctc_logits.pt")
                except Exception as e:
                    print(f"Failed to save CTC logits: {e}")
                    print(f"CTC logit types: {[type(x) for x in self.logit_storage['ctc_logits'][:3]]}")
                    return False
            
            if self.logit_storage['decoder_logits']:
                try:
                    print(f"Concatenating {len(self.logit_storage['decoder_logits'])} decoder logits")
                    decoder_logits = torch.cat(self.logit_storage['decoder_logits'], dim=0)
                    print(f"Decoder logits concatenated: {decoder_logits.shape}")
                    torch.save(decoder_logits, f"{output_dir}/{model_name}_decoder_logits.pt")
                except Exception as e:
                    print(f"Failed to save decoder logits: {e}")
                    print(f"Decoder logit types: {[type(x) for x in self.logit_storage['decoder_logits'][:3]]}")
                    return False
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'num_utterances': len(self.logit_storage['utterance_ids']),
                'vocab_size': ctc_logits.shape[-1] if self.logit_storage['ctc_logits'] else None,
                'utterance_ids': self.logit_storage['utterance_ids'],
                'wav_lens': [item.tolist() if torch.is_tensor(item) else item for item in self.logit_storage['wav_lens']],
                'token_lens': [item.tolist() if torch.is_tensor(item) else item for item in self.logit_storage['token_lens']],
                'target_words': self.logit_storage['target_words']
            }
            
            with open(f"{output_dir}/{model_name}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Successfully saved logits for {model_name}")
            return True
            
        except Exception as e:
            print(f"Unexpected error in save_extracted_logits for {model_name}: {e}")
            import traceback
            print(traceback.format_exc())
            return False

def test_logit_storage():
    """Test if our logit storage is working correctly"""
    print("Testing logit storage mechanism...")
    
    # Initialize test storage
    storage = TestLogitStorage()
    
    print("Storage initialized")
    print(f"Initial storage state: {[(k, len(v) if hasattr(v, '__len__') else type(v)) for k, v in storage.logit_storage.items()]}")
    
    # Create a dummy batch for testing
    batch_size = 2
    seq_len = 100
    vocab_size = 1000
    
    # Simulate what happens in compute_forward during TEST stage
    ctc_logits = torch.randn(batch_size, seq_len, vocab_size)
    decoder_logits = torch.randn(batch_size, 20, vocab_size)
    wav_lens = torch.tensor([1.0, 0.8])
    token_lens = torch.tensor([15, 12])
    utterance_ids = ["test_001", "test_002"]
    target_words = ["hello world", "test phrase"]
    
    print("\nSimulating logit storage...")
    
    # Store logits as would happen in compute_forward
    storage.logit_storage['ctc_logits'].append(ctc_logits.detach().cpu())
    storage.logit_storage['decoder_logits'].append(decoder_logits.detach().cpu())
    storage.logit_storage['wav_lens'].append(wav_lens.detach().cpu())
    storage.logit_storage['token_lens'].append(token_lens.detach().cpu())
    storage.logit_storage['utterance_ids'].extend(utterance_ids)
    storage.logit_storage['target_words'].extend(target_words)
    
    print(f"After storage: {[(k, len(v) if hasattr(v, '__len__') else type(v)) for k, v in storage.logit_storage.items()]}")
    
    # Test saving
    print("\nTesting save functionality...")
    output_dir = "/tmp/test_logits"
    os.makedirs(output_dir, exist_ok=True)
    
    success = storage.save_extracted_logits(output_dir, "test_model")
    print(f"Save result: {success}")
    
    if success:
        # Check what files were created
        files = os.listdir(output_dir)
        print(f"Created files: {files}")
        
        # Check file sizes
        for f in files:
            path = os.path.join(output_dir, f)
            size = os.path.getsize(path)
            print(f"  {f}: {size} bytes")

if __name__ == "__main__":
    test_logit_storage()