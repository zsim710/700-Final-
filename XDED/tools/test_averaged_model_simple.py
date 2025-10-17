#!/usr/bin/env python3
"""
Simplified inference script that loads averaged SA models directly
without using SpeechBrain's pretrainer to avoid symlink issues.
"""

import os
import sys
import json
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

# Add paths for SpeechBrain
sys.path.insert(0, '/home/zsim710/XDED/conformer/conformer-asr')

import speechbrain as sb
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
from speechbrain.lobes.models.convolution import ConvolutionFrontEnd
from speechbrain.nnet.linear import Linear
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.features import Fbank
from speechbrain.dataio.dataio import read_audio
import sentencepiece
from jiwer import wer

# Test CSV paths (corrected)
TEST_CSV_PATHS = {
    'M01': '/home/zsim710/partitions/uaspeech/by_speakers/M01.csv',
    'M05': '/home/zsim710/partitions/uaspeech/by_speakers/M05.csv',
    'M08': '/home/zsim710/partitions/uaspeech/by_speakers/M08.csv',
    'M16': '/home/zsim710/partitions/uaspeech/by_speakers/M16.csv',
}

def load_model_components(checkpoint_dir, device='cuda', donor_head_ckpt: str = None):
    """
    Load model components directly without using pretrainer.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        device: Device to load on
        
    Returns:
        Tuple of (modules dict, tokenizer, normalizer, compute_features)
    """
    print(f"Loading model components from: {checkpoint_dir}")
    
    # Load model weights
    model_path = os.path.join(checkpoint_dir, 'model.ckpt')
    print(f"Loading model weights from: {model_path}")
    model_state = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model architecture
    print("Creating model architecture...")
    
    # CNN frontend - CORRECT SA model architecture!
    cnn = ConvolutionFrontEnd(
        input_shape=(8, 10, 80),
        num_blocks=2,  # SA models use 2, not 3!
        num_layers_per_block=1,
        out_channels=(64, 32),  # SA models use (64, 32), not (64, 64, 64)!
        kernel_sizes=(3, 3),  # SA models use (3, 3), not (5, 5, 1)!
        strides=(2, 2),
        residuals=(False, False)
    ).to(device)
    
    # Transformer - CORRECT SA model architecture!
    transformer = TransformerASR(
        input_size=640,  # 20 freq bins × 32 channels = 640, not 1280!
        tgt_vocab=5000,
        d_model=144,  # SA models use 144, not 512!
        nhead=4,
        num_encoder_layers=12,
        num_decoder_layers=4,  # SA models use 4, not 6!
        d_ffn=1024,  # SA models use 1024, not 2048!
        dropout=0.1,
        activation=torch.nn.GELU,
        encoder_module='transformer',
        attention_type='regularMHA',
        normalize_before=True,
        causal=False
    ).to(device)
    
    # Output layers - CORRECT dimensions!
    ctc_lin = Linear(input_size=144, n_neurons=5000).to(device)  # d_model=144
    seq_lin = Linear(input_size=144, n_neurons=5000).to(device)  # d_model=144
    
    # Load weights into modules
    print("Loading weights into modules...")
    
    # Extract model state - the averaged checkpoint has 'model' key
    if isinstance(model_state, dict) and 'model' in model_state:
        print(f"Found 'model' key, extracting state dict...")
        print(f"Held out: {model_state.get('held_out', 'unknown')}")
        print(f"Num models averaged: {model_state.get('num_models_averaged', 'unknown')}")
        actual_state = model_state['model']
    else:
        actual_state = model_state
    
    # Map numbered prefixes to modules: 0=CNN, 1=Transformer, 2=CTC, 3=Seq
    cnn_state = {k.replace('0.', ''): v for k, v in actual_state.items() if k.startswith('0.')}
    transformer_state = {k.replace('1.', ''): v for k, v in actual_state.items() if k.startswith('1.')}
    ctc_state = {k.replace('2.', ''): v for k, v in actual_state.items() if k.startswith('2.')}
    seq_state = {k.replace('3.', ''): v for k, v in actual_state.items() if k.startswith('3.')}

    # Optionally override CTC/Seq heads from a donor SA checkpoint (encoder-only averaging)
    if donor_head_ckpt is not None and os.path.isfile(donor_head_ckpt):
        print(f"Overriding CTC/Seq heads from donor checkpoint: {donor_head_ckpt}")
        donor_sd = torch.load(donor_head_ckpt, map_location=device, weights_only=False)
        donor_ctc = {k.replace('2.', ''): v for k, v in donor_sd.items() if k.startswith('2.')}
        donor_seq = {k.replace('3.', ''): v for k, v in donor_sd.items() if k.startswith('3.')}
        if len(donor_ctc) == 0:
            print("Warning: Donor checkpoint missing CTC head (prefix '2.'). Keeping averaged CTC head.")
        else:
            ctc_state = donor_ctc
        if len(donor_seq) == 0:
            print("Warning: Donor checkpoint missing Seq head (prefix '3.'). Keeping averaged Seq head.")
        else:
            seq_state = donor_seq
    
    print(f"Loading {len(cnn_state)} CNN parameters")
    print(f"Loading {len(transformer_state)} Transformer parameters")
    print(f"Loading {len(ctc_state)} CTC parameters")
    print(f"Loading {len(seq_state)} Seq parameters")
    
    cnn.load_state_dict(cnn_state, strict=False)
    transformer.load_state_dict(transformer_state, strict=False)
    ctc_lin.load_state_dict(ctc_state, strict=False)
    seq_lin.load_state_dict(seq_state, strict=False)
    
    # Set to eval mode
    cnn.eval()
    transformer.eval()
    ctc_lin.eval()
    seq_lin.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer')
    tokenizer.load(tokenizer_path)
    
    # Load normalizer
    print("Loading normalizer...")
    normalizer_path = os.path.join(checkpoint_dir, 'normalizer.ckpt')
    normalizer = None
    if os.path.isfile(normalizer_path):
        try:
            normalizer_state = torch.load(normalizer_path, map_location=device, weights_only=False)
            normalizer = InputNormalization(norm_type='global')
            normalizer.load_state_dict(normalizer_state, strict=False)
            normalizer.to(device)
            normalizer.eval()
        except Exception as e:
            print(f"Warning: Failed to load normalizer from {normalizer_path}: {e}. Falling back to simple per-utterance norm.")
            normalizer = None
    else:
        print(f"Warning: normalizer not found at {normalizer_path}. Falling back to simple per-utterance norm.")
    
    # Create feature extractor
    compute_features = Fbank(sample_rate=16000, n_fft=400, n_mels=80)
    
    modules = {
        'cnn': cnn,
        'transformer': transformer,
        'ctc_lin': ctc_lin,
        'seq_lin': seq_lin
    }
    
    print("✓ Model loaded successfully")
    return modules, tokenizer, normalizer, compute_features

def transcribe_file(audio_path, modules, tokenizer, normalizer, compute_features, device='cuda', simple_norm: bool = False):
    """
    Transcribe a single audio file.
    
    Args:
        audio_path: Path to audio file
        modules: Dict of model modules
        tokenizer: SentencePiece tokenizer
        normalizer: Feature normalizer
        compute_features: Feature extractor
        device: Device for computation
        
    Returns:
        Transcription string
    """
    # Load audio
    signal = read_audio(audio_path)
    signal = signal.to(device).unsqueeze(0)  # Add batch dimension
    
    # Extract features
    with torch.no_grad():
        feats = compute_features(signal)
        if normalizer is not None and not simple_norm:
            feats = normalizer(feats, torch.tensor([1.0]).to(device))
        else:
            # Simple per-utterance normalization (mean/var over time/freq)
            mean = feats.mean(dim=(1, 2), keepdim=True)
            std = feats.std(dim=(1, 2), keepdim=True) + 1e-5
            feats = (feats - mean) / std
        
        # Forward through CNN
        src = modules['cnn'](feats)
        
        # Forward through transformer encoder
        # Use the encode method for encoder-only inference
        enc_out = modules['transformer'].encode(src)
        
        # Greedy decoding using CTC
        logits = modules['ctc_lin'](enc_out)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get best path
        predicted_tokens = torch.argmax(log_probs, dim=-1).squeeze(0)
        
        # Remove blanks and repeated tokens (CTC decoding)
        decoded_tokens = []
        prev_token = None
        for token in predicted_tokens:
            token = token.item()
            if token != 0 and token != prev_token:  # 0 is blank
                decoded_tokens.append(token)
            prev_token = token
        
        # Decode tokens to text
        if len(decoded_tokens) > 0:
            transcription = tokenizer.decode(decoded_tokens)
        else:
            transcription = ""
    
    return transcription

def test_model_on_speaker(checkpoint_dir, test_csv, speaker_id, output_file=None, device='cuda', donor_head_ckpt: str = None, simple_norm: bool = False):
    """
    Test a model on a speaker's test data.
    
    Args:
        checkpoint_dir: Path to model checkpoint directory
        test_csv: Path to test CSV file
        speaker_id: Speaker ID being tested
        output_file: Optional path to save results
        device: Device for inference
        
    Returns:
        Dict with evaluation results
    """
    print(f"\nTesting model on speaker {speaker_id}...")
    print(f"Test data: {test_csv}")
    
    # Load model
    modules, tokenizer, normalizer, compute_features = load_model_components(checkpoint_dir, device, donor_head_ckpt=donor_head_ckpt)
    
    # Load test data
    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} test utterances")
    
    # Run inference
    predictions = []
    references = []
    results_details = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        audio_path = row['wav']
        reference = row['wrd'].strip().upper()
        
        try:
            # Transcribe
            prediction = transcribe_file(
                audio_path, modules, tokenizer, normalizer, compute_features, device, simple_norm=simple_norm
            )
            prediction = prediction.strip().upper()
            
            predictions.append(prediction)
            references.append(reference)
            
            results_details.append({
                'id': row['ID'],
                'audio_path': audio_path,
                'reference': reference,
                'prediction': prediction,
            })
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {str(e)}")
            predictions.append("")
            references.append(reference)
            results_details.append({
                'id': row['ID'],
                'audio_path': audio_path,
                'reference': reference,
                'prediction': "",
                'error': str(e)
            })
    
    # Compute WER
    error_rate = wer(references, predictions)
    wer_percentage = error_rate * 100.0
    
    # Create results
    results = {
        'speaker_id': speaker_id,
        'num_utterances': len(df),
        'wer': wer_percentage,
        'accuracy': 100.0 - wer_percentage,
        'details': results_details
    }
    
    print(f"✓ WER: {wer_percentage:.2f}%")
    print(f"✓ Accuracy: {results['accuracy']:.2f}%")
    
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test averaged SA model on held-out speaker")
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--speaker_id', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--donor_head_ckpt', type=str, default=None, help='Path to SA checkpoint to use CTC/Seq heads from (encoder-only averaging)')
    parser.add_argument('--simple_norm', action='store_true', help='Use simple per-utterance normalization instead of checkpoint normalizer')
    args = parser.parse_args()
    
    # Test model
    results = test_model_on_speaker(
        args.checkpoint_dir,
        args.test_csv,
        args.speaker_id,
        args.output_file,
        args.device,
        donor_head_ckpt=args.donor_head_ckpt,
        simple_norm=args.simple_norm
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Speaker: {args.speaker_id}")
    print(f"WER: {results['wer']:.2f}%")
    print(f"Accuracy: {results['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
