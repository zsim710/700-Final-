#!/usr/bin/env python3
"""
Verify extracted logits by loading them and checking if they decode correctly.
This script loads the saved logits and compares the decoded outputs with the original targets.
"""

import os
import torch
import json
import numpy as np
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

# Set GPU and disable wandb
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['WANDB_MODE'] = 'disabled'

def load_logits_and_metadata(model_name, base_dir="/home/zsim710/XDED/speechbrain/exp_results/logit_extraction"):
    """Load extracted logits and metadata for a specific model."""
    
    model_dir = os.path.join(base_dir, model_name, model_name)
    
    # Load files
    ctc_logits_file = f"{model_dir}/{model_name}_ctc_logits.pt"
    decoder_logits_file = f"{model_dir}/{model_name}_decoder_logits.pt"
    metadata_file = f"{model_dir}/{model_name}_metadata.json"
    
    print(f"Loading logits for {model_name}...")
    print(f"  CTC logits: {ctc_logits_file}")
    print(f"  Decoder logits: {decoder_logits_file}")
    print(f"  Metadata: {metadata_file}")
    
    # Check if files exist
    for file_path in [ctc_logits_file, decoder_logits_file, metadata_file]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
    
    try:
        # Load logits
        ctc_logits = torch.load(ctc_logits_file, map_location='cpu')
        decoder_logits = torch.load(decoder_logits_file, map_location='cpu')
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Successfully loaded logits for {model_name}")
        print(f"  Number of utterances: {metadata['num_utterances']}")
        print(f"  Vocabulary size: {metadata['vocab_size']}")
        
        if isinstance(ctc_logits, list):
            print(f"  CTC logits: List of {len(ctc_logits)} tensors (variable length)")
            print(f"  CTC shapes (first 3): {[x.shape for x in ctc_logits[:3]]}")
        else:
            print(f"  CTC logits shape: {ctc_logits.shape}")
            
        if isinstance(decoder_logits, list):
            print(f"  Decoder logits: List of {len(decoder_logits)} tensors (variable length)")
            print(f"  Decoder shapes (first 3): {[x.shape for x in decoder_logits[:3]]}")
        else:
            print(f"  Decoder logits shape: {decoder_logits.shape}")
        
        return {
            'ctc_logits': ctc_logits,
            'decoder_logits': decoder_logits,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"❌ Error loading logits for {model_name}: {e}")
        return None

def load_tokenizer(hparams_file="hparams/exp/uaspeech/ua_SA_val_uncommon_WRA.yaml"):
    """Load the tokenizer from the hyperparameters file."""
    
    print(f"Loading tokenizer from {hparams_file}...")
    
    try:
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, {})
        
        # Load pretrained tokenizer
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()
        
        tokenizer = hparams["tokenizer"]
        print(f"✅ Tokenizer loaded successfully")
        return tokenizer
        
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return None

def extract_ground_truth_tokens(metadata, tokenizer):
    """Extract ground truth tokens from target words using the tokenizer."""
    
    target_words = metadata['target_words']
    ground_truth_tokens = []
    
    print(f"Extracting ground truth tokens from {len(target_words)} utterances...")
    
    for i, word_text in enumerate(target_words):
        try:
            # Encode the text to get token IDs
            token_ids = tokenizer.encode_as_ids(word_text)
            ground_truth_tokens.append(token_ids)
        except Exception as e:
            print(f"Warning: Failed to tokenize utterance {i}: '{word_text}' - {e}")
            ground_truth_tokens.append([])
    
    return ground_truth_tokens

def decode_logits_to_tokens(logits, is_ctc=False):
    """Decode logits to token IDs only."""
    
    predicted_tokens = []
    
    if isinstance(logits, list):
        # Handle variable-length logits (list format)
        for i, logit_tensor in enumerate(logits):
            if len(logit_tensor.shape) == 2:  # [T, V] or [U, V]
                if is_ctc:
                    # For CTC, we need to handle repeated tokens and blanks
                    pred_token_ids = torch.argmax(logit_tensor, dim=-1)  # [T]
                    # Simple CTC decoding (remove blanks and repeated tokens)
                    # Blank token is usually index 0
                    filtered_tokens = []
                    prev_token = None
                    for token in pred_token_ids:
                        token_val = token.item()
                        if token_val != 0 and token_val != prev_token:  # Not blank and not repeated
                            filtered_tokens.append(token_val)
                        prev_token = token_val
                    predicted_tokens.append(filtered_tokens)
                else:
                    # For seq2seq decoder, just take argmax
                    pred_token_ids = torch.argmax(logit_tensor, dim=-1)  # [U]
                    # Remove EOS token (index 2) and everything after
                    tokens_list = pred_token_ids.tolist()
                    if 2 in tokens_list:
                        eos_idx = tokens_list.index(2)
                        tokens_list = tokens_list[:eos_idx]
                    predicted_tokens.append(tokens_list)
    
    else:
        # Handle concatenated logits (tensor format)
        if is_ctc:
            # CTC decoding for concatenated tensor [B, T, V]
            pred_token_ids = torch.argmax(logits, dim=-1)  # [B, T]
            
            for i in range(pred_token_ids.shape[0]):
                # Simple CTC decoding for each utterance
                utt_tokens = pred_token_ids[i]  # [T]
                filtered_tokens = []
                prev_token = None
                for token in utt_tokens:
                    token_val = token.item()
                    if token_val != 0 and token_val != prev_token:
                        filtered_tokens.append(token_val)
                    prev_token = token_val
                
                predicted_tokens.append(filtered_tokens)
        else:
            # Seq2seq decoding for concatenated tensor [B, U, V]
            pred_token_ids = torch.argmax(logits, dim=-1)  # [B, U]
            
            for i in range(pred_token_ids.shape[0]):
                utt_tokens = pred_token_ids[i].tolist()  # [U]
                # Remove EOS and everything after
                if 2 in utt_tokens:
                    eos_idx = utt_tokens.index(2)
                    utt_tokens = utt_tokens[:eos_idx]
                
                predicted_tokens.append(utt_tokens)
    
    return predicted_tokens

def decode_logits(logits, tokenizer, is_ctc=False):
    """Decode logits to tokens and text."""
    
    if isinstance(logits, list):
        # Handle variable-length logits (list format)
        decoded_tokens = []
        decoded_text = []
        
        for i, logit_tensor in enumerate(logits):
            if len(logit_tensor.shape) == 2:  # [T, V] or [U, V]
                # Get most likely tokens
                if is_ctc:
                    # For CTC, we need to handle repeated tokens and blanks
                    pred_tokens = torch.argmax(logit_tensor, dim=-1)  # [T]
                    # Simple CTC decoding (remove blanks and repeated tokens)
                    # Blank token is usually index 0
                    filtered_tokens = []
                    prev_token = None
                    for token in pred_tokens:
                        token_val = token.item()
                        if token_val != 0 and token_val != prev_token:  # Not blank and not repeated
                            filtered_tokens.append(token_val)
                        prev_token = token_val
                    decoded_tokens.append(filtered_tokens)
                else:
                    # For seq2seq decoder, just take argmax
                    pred_tokens = torch.argmax(logit_tensor, dim=-1)  # [U]
                    # Remove EOS token (index 2) and everything after
                    tokens_list = pred_tokens.tolist()
                    if 2 in tokens_list:
                        eos_idx = tokens_list.index(2)
                        tokens_list = tokens_list[:eos_idx]
                    decoded_tokens.append(tokens_list)
                
                # Decode to text
                try:
                    text = tokenizer.decode_ids(decoded_tokens[-1])
                    decoded_text.append(text)
                except Exception as e:
                    print(f"Warning: Failed to decode utterance {i}: {e}")
                    decoded_text.append("")
    
    else:
        # Handle concatenated logits (tensor format)
        if is_ctc:
            # CTC decoding for concatenated tensor [B, T, V]
            pred_tokens = torch.argmax(logits, dim=-1)  # [B, T]
            decoded_tokens = []
            decoded_text = []
            
            for i in range(pred_tokens.shape[0]):
                # Simple CTC decoding for each utterance
                utt_tokens = pred_tokens[i]  # [T]
                filtered_tokens = []
                prev_token = None
                for token in utt_tokens:
                    token_val = token.item()
                    if token_val != 0 and token_val != prev_token:
                        filtered_tokens.append(token_val)
                    prev_token = token_val
                
                decoded_tokens.append(filtered_tokens)
                try:
                    text = tokenizer.decode_ids(filtered_tokens)
                    decoded_text.append(text)
                except:
                    decoded_text.append("")
        else:
            # Seq2seq decoding for concatenated tensor [B, U, V]
            pred_tokens = torch.argmax(logits, dim=-1)  # [B, U]
            decoded_tokens = []
            decoded_text = []
            
            for i in range(pred_tokens.shape[0]):
                utt_tokens = pred_tokens[i].tolist()  # [U]
                # Remove EOS and everything after
                if 2 in utt_tokens:
                    eos_idx = utt_tokens.index(2)
                    utt_tokens = utt_tokens[:eos_idx]
                
                decoded_tokens.append(utt_tokens)
                try:
                    text = tokenizer.decode_ids(utt_tokens)
                    decoded_text.append(text)
                except:
                    decoded_text.append("")
    
    return decoded_tokens, decoded_text

def compare_token_sequences(predicted_tokens, ground_truth_tokens, utterance_ids, logit_type=""):
    """Compare predicted and ground truth token sequences."""
    
    print(f"\n🔍 TOKEN-LEVEL COMPARISON ({logit_type}):")
    print(f"Total utterances: {len(ground_truth_tokens)}")
    
    # Statistics
    exact_matches = 0
    total_tokens_gt = 0
    total_tokens_pred = 0
    token_matches = 0
    length_matches = 0
    
    # Detailed comparison for first few examples
    print(f"\n📝 Sample token comparisons (first 10 utterances):")
    
    for i in range(min(10, len(ground_truth_tokens))):
        gt_tokens = ground_truth_tokens[i]
        pred_tokens = predicted_tokens[i] if i < len(predicted_tokens) else []
        utt_id = utterance_ids[i] if i < len(utterance_ids) else f"utt_{i}"
        
        # Check exact match
        is_exact_match = gt_tokens == pred_tokens
        if is_exact_match:
            exact_matches += 1
        
        # Check length match
        if len(gt_tokens) == len(pred_tokens):
            length_matches += 1
        
        # Count token-level matches
        total_tokens_gt += len(gt_tokens)
        total_tokens_pred += len(pred_tokens)
        
        # Count individual token matches (alignment-based)
        min_len = min(len(gt_tokens), len(pred_tokens))
        for j in range(min_len):
            if gt_tokens[j] == pred_tokens[j]:
                token_matches += 1
        
        # Print detailed comparison
        print(f"\nUtterance {i+1}: {utt_id}")
        print(f"  GT tokens:   {gt_tokens} (len={len(gt_tokens)})")
        print(f"  Pred tokens: {pred_tokens} (len={len(pred_tokens)})")
        print(f"  Exact match: {'✅' if is_exact_match else '❌'}")
        
        # Show token-by-token comparison for mismatches
        if not is_exact_match and i < 5:  # Only show detailed for first 5 mismatches
            match_str = ""
            for j in range(max(len(gt_tokens), len(pred_tokens))):
                gt_tok = gt_tokens[j] if j < len(gt_tokens) else None
                pred_tok = pred_tokens[j] if j < len(pred_tokens) else None
                
                if gt_tok == pred_tok:
                    match_str += "✅"
                else:
                    match_str += f"❌({gt_tok}→{pred_tok})"
                match_str += " "
            print(f"  Token match: {match_str}")
    
    # Overall statistics
    print(f"\n📊 TOKEN-LEVEL STATISTICS ({logit_type}):")
    print(f"  Exact sequence matches: {exact_matches}/{len(ground_truth_tokens)} ({100*exact_matches/len(ground_truth_tokens):.1f}%)")
    print(f"  Length matches: {length_matches}/{len(ground_truth_tokens)} ({100*length_matches/len(ground_truth_tokens):.1f}%)")
    print(f"  Total GT tokens: {total_tokens_gt}")
    print(f"  Total predicted tokens: {total_tokens_pred}")
    if total_tokens_gt > 0:
        print(f"  Token-level accuracy: {token_matches}/{total_tokens_gt} ({100*token_matches/total_tokens_gt:.1f}%)")
    
    return {
        'exact_matches': exact_matches,
        'length_matches': length_matches,
        'token_accuracy': token_matches / total_tokens_gt if total_tokens_gt > 0 else 0,
        'total_utterances': len(ground_truth_tokens)
    }

def verify_model_logits(model_name, tokenizer, base_dir="/home/zsim710/XDED/speechbrain/exp_results/logit_extraction"):
    """Verify logits for a specific model by decoding and comparing with targets."""
    
    print(f"\n{'='*60}")
    print(f"VERIFYING LOGITS FOR {model_name}")
    print(f"{'='*60}")
    
    # Load logits and metadata
    data = load_logits_and_metadata(model_name, base_dir)
    if data is None:
        return False
    
    ctc_logits = data['ctc_logits']
    decoder_logits = data['decoder_logits']
    metadata = data['metadata']
    
    # Decode CTC logits
    print(f"\n🔍 Decoding CTC logits...")
    ctc_tokens, ctc_text = decode_logits(ctc_logits, tokenizer, is_ctc=True)
    
    # Decode decoder logits  
    print(f"🔍 Decoding decoder logits...")
    decoder_tokens, decoder_text = decode_logits(decoder_logits, tokenizer, is_ctc=False)
    
    # Compare with ground truth
    target_words = metadata['target_words']
    utterance_ids = metadata['utterance_ids']
    
    print(f"\n📊 VERIFICATION RESULTS:")
    print(f"Total utterances: {len(target_words)}")
    
    # Show first 5 examples
    print(f"\n📝 Sample comparisons (first 5 utterances):")
    for i in range(min(5, len(target_words))):
        print(f"\nUtterance {i+1}: {utterance_ids[i]}")
        print(f"  Ground truth: '{target_words[i]}'")
        print(f"  CTC decoded:  '{ctc_text[i] if i < len(ctc_text) else 'N/A'}'")
        print(f"  Decoder decoded: '{decoder_text[i] if i < len(decoder_text) else 'N/A'}'")
    
    # Calculate some basic statistics
    valid_ctc = sum(1 for text in ctc_text if text.strip())
    valid_decoder = sum(1 for text in decoder_text if text.strip())
    
    print(f"\n📈 Statistics:")
    print(f"  CTC non-empty decodings: {valid_ctc}/{len(ctc_text)} ({100*valid_ctc/len(ctc_text):.1f}%)")
    print(f"  Decoder non-empty decodings: {valid_decoder}/{len(decoder_text)} ({100*valid_decoder/len(decoder_text):.1f}%)")
    
    # Simple word-level accuracy check (just for decoder since it's more reliable)
    if len(decoder_text) == len(target_words):
        exact_matches = sum(1 for pred, target in zip(decoder_text, target_words) if pred.strip() == target.strip())
        print(f"  Decoder exact matches: {exact_matches}/{len(target_words)} ({100*exact_matches/len(target_words):.1f}%)")
    
    return True

def verify_model_tokens(model_name, tokenizer, base_dir="/home/zsim710/XDED/speechbrain/exp_results/logit_extraction"):
    """Verify logits by comparing token-level predictions with ground truth."""
    
    print(f"\n🎯 TOKEN-LEVEL VERIFICATION for {model_name}")
    print("=" * 60)
    
    # Load logit data using the correct file format
    data = load_logits_and_metadata(model_name, base_dir)
    if data is None:
        print(f"❌ Failed to load logits for {model_name}")
        return
    
    # Convert to the format expected by our token functions
    logit_data = []
    metadata = data['metadata']
    
    # Reconstruct logit_data format from loaded data
    ctc_logits = data['ctc_logits']
    decoder_logits = data['decoder_logits']
    
    # Create the expected data structure
    if isinstance(ctc_logits, list):
        # Individual utterance format
        for i in range(len(metadata['utterance_ids'])):
            item = {
                'id': metadata['utterance_ids'][i],
                'targets': metadata['target_words'][i],
                'ctc_logits': ctc_logits[i] if i < len(ctc_logits) else None,
                'decoder_logits': decoder_logits[i] if i < len(decoder_logits) else None
            }
            logit_data.append(item)
    else:
        # Handle tensor format - this is more complex, let's use metadata info
        for i in range(len(metadata['utterance_ids'])):
            item = {
                'id': metadata['utterance_ids'][i],
                'targets': metadata['target_words'][i]
            }
            logit_data.append(item)
    
    print(f"📦 Loaded {len(logit_data)} utterances")
    
    # Extract ground truth tokens
    print("\n🔤 Extracting ground truth tokens...")
    ground_truth_tokens = extract_ground_truth_tokens(metadata, tokenizer)
    utterance_ids = metadata['utterance_ids']
    
    print(f"✅ Extracted tokens for {len(ground_truth_tokens)} utterances")
    
    # Decode logits to tokens for both CTC and decoder  
    print("\n🔄 Decoding CTC logits to tokens...")
    ctc_tokens = decode_logits_to_tokens(data['ctc_logits'], is_ctc=True)
    
    print("🔄 Decoding decoder logits to tokens...")
    decoder_tokens = decode_logits_to_tokens(data['decoder_logits'], is_ctc=False)
    
    # Compare token sequences
    print("\n" + "="*60)
    ctc_stats = compare_token_sequences(ctc_tokens, ground_truth_tokens, utterance_ids, "CTC")
    
    print("\n" + "="*60)
    decoder_stats = compare_token_sequences(decoder_tokens, ground_truth_tokens, utterance_ids, "DECODER")
    
    # Final summary
    print("\n" + "="*80)
    print(f"🏆 FINAL TOKEN-LEVEL SUMMARY for {model_name}")
    print("="*80)
    print(f"CTC Token Results:")
    print(f"  - Exact sequence matches: {ctc_stats['exact_matches']}/{ctc_stats['total_utterances']} ({100*ctc_stats['exact_matches']/ctc_stats['total_utterances']:.1f}%)")
    print(f"  - Token-level accuracy: {100*ctc_stats['token_accuracy']:.1f}%")
    
    print(f"\nDecoder Token Results:")
    print(f"  - Exact sequence matches: {decoder_stats['exact_matches']}/{decoder_stats['total_utterances']} ({100*decoder_stats['exact_matches']/decoder_stats['total_utterances']:.1f}%)")
    print(f"  - Token-level accuracy: {100*decoder_stats['token_accuracy']:.1f}%")
    
    return {
        'model': model_name,
        'ctc_stats': ctc_stats,
        'decoder_stats': decoder_stats
    }

def main():
    """Main verification function."""
    
    print("🔬 LOGIT VERIFICATION SCRIPT")
    print("="*60)
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    if tokenizer is None:
        print("❌ Failed to load tokenizer. Cannot proceed with verification.")
        return
    
    # Check which models have extracted logits
    base_dir = "/home/zsim710/XDED/speechbrain/exp_results/logit_extraction"
    
    if not os.path.exists(base_dir):
        print(f"❌ Logit extraction directory not found: {base_dir}")
        return
    
    # Find available models
    available_models = []
    for item in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, item)
        if os.path.isdir(model_dir):
            # Check if it has the required files
            model_subdir = os.path.join(model_dir, item)
            if os.path.exists(f"{model_subdir}/{item}_metadata.json"):
                available_models.append(item)
    
    print(f"📁 Found {len(available_models)} models with extracted logits:")
    for model in available_models:
        print(f"  - {model}")
    
    if not available_models:
        print("❌ No models with extracted logits found.")
        return
    
    # Verify each model
    successful_verifications = []
    failed_verifications = []
    
    for model in available_models:
        try:
            success = verify_model_logits(model, tokenizer, base_dir)
            if success:
                successful_verifications.append(model)
            else:
                failed_verifications.append(model)
        except Exception as e:
            print(f"❌ Exception during verification of {model}: {e}")
            failed_verifications.append(model)
    
    # Final summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total models checked: {len(available_models)}")
    print(f"Successful verifications: {len(successful_verifications)}")
    print(f"Failed verifications: {len(failed_verifications)}")
    
    if successful_verifications:
        print(f"\n✅ Successfully verified:")
        for model in successful_verifications:
            print(f"  - {model}")
    
    if failed_verifications:
        print(f"\n❌ Failed to verify:")
        for model in failed_verifications:
            print(f"  - {model}")

if __name__ == "__main__":
    main()