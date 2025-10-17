#!/usr/bin/env python3
"""Simple test to check logit storage mechanism"""

import os
import sys
sys.path.insert(0, '/home/zsim710/XDED/conformer/conformer-asr/')

import torch
import json
from extract_sa_logits import LogitExtractorASR
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

def test_logit_storage():
    """Test if our logit storage is working correctly"""
    print("Testing logit storage mechanism...")
    
    # Load F03 model configuration
    with open("hparams/train.yaml") as fin:
        hparams = load_hyperpyyaml(fin, {})
    
    # Set F03 model paths
    model_name = "F03"
    checkpoint_path = f"/home/zsim710/results/uaspeech/{model_name}/M12/7775/save"
    hparams["pretrainer"].set_collect_in(checkpoint_path)
    
    # Initialize model
    asr_brain = LogitExtractorASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts={'device': 'cuda:0'},
        checkpointer=hparams["checkpointer"],
    )
    
    print("Model initialized")
    print(f"Initial storage state: {[(k, len(v) if hasattr(v, '__len__') else type(v)) for k, v in asr_brain.logit_storage.items()]}")
    
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
    asr_brain.logit_storage['ctc_logits'].append(ctc_logits.detach().cpu())
    asr_brain.logit_storage['decoder_logits'].append(decoder_logits.detach().cpu())
    asr_brain.logit_storage['wav_lens'].append(wav_lens.detach().cpu())
    asr_brain.logit_storage['token_lens'].append(token_lens.detach().cpu())
    asr_brain.logit_storage['utterance_ids'].extend(utterance_ids)
    asr_brain.logit_storage['target_words'].extend(target_words)
    
    print(f"After storage: {[(k, len(v) if hasattr(v, '__len__') else type(v)) for k, v in asr_brain.logit_storage.items()]}")
    
    # Test concatenation
    print("\nTesting concatenation...")
    
    if asr_brain.logit_storage['ctc_logits']:
        try:
            ctc_concat = torch.cat(asr_brain.logit_storage['ctc_logits'], dim=0)
            print(f"CTC concatenation successful: {ctc_concat.shape}")
        except Exception as e:
            print(f"CTC concatenation failed: {e}")
    
    if asr_brain.logit_storage['decoder_logits']:
        try:
            decoder_concat = torch.cat(asr_brain.logit_storage['decoder_logits'], dim=0)
            print(f"Decoder concatenation successful: {decoder_concat.shape}")
        except Exception as e:
            print(f"Decoder concatenation failed: {e}")
    
    # Test saving
    print("\nTesting save functionality...")
    output_dir = "/tmp/test_logits"
    os.makedirs(output_dir, exist_ok=True)
    
    success = asr_brain.save_extracted_logits(output_dir, "test_model")
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