#!/usr/bin/env python3
"""
Inference script to test SA models (both averaged and individual) on held-out speakers.
Uses SpeechBrain's EncoderDecoderASR interface for transcription.
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

from speechbrain.inference.ASR import EncoderDecoderASR
from jiwer import wer

def load_asr_model(checkpoint_dir, device='cuda'):
    """
    Load a SpeechBrain ASR model from a checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory containing model.ckpt, hyperparams.yaml, etc.
        device: Device to load model on
        
    Returns:
        EncoderDecoderASR model ready for inference
    """
    print(f"Loading ASR model from: {checkpoint_dir}")
    
    # Load using SpeechBrain's EncoderDecoderASR interface
    asr_model = EncoderDecoderASR.from_hparams(
        source=checkpoint_dir,
        savedir=checkpoint_dir,
        run_opts={"device": device}
    )
    
    print(f"✓ Model loaded successfully")
    return asr_model

def test_model_on_speaker(asr_model, test_csv, speaker_id, output_file=None):
    """
    Test an ASR model on a speaker's test data and compute WER.
    
    Args:
        asr_model: Loaded SpeechBrain ASR model
        test_csv: Path to test CSV file
        speaker_id: Speaker ID being tested
        output_file: Optional path to save detailed results
        
    Returns:
        Dict with evaluation results
    """
    print(f"\nTesting model on speaker {speaker_id}...")
    print(f"Test data: {test_csv}")
    
    # Load test data
    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} test utterances")
    
    # Run inference on all utterances
    predictions = []
    references = []
    results_details = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        audio_path = row['wav']
        reference = row['wrd'].strip().upper()
        
        try:
            # Transcribe the audio file
            prediction = asr_model.transcribe_file(audio_path)
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
    
    # Create results summary
    results = {
        'speaker_id': speaker_id,
        'num_utterances': len(df),
        'wer': wer_percentage,
        'accuracy': 100.0 - wer_percentage,
        'details': results_details
    }
    
    print(f"✓ WER: {wer_percentage:.2f}%")
    print(f"✓ Accuracy: {results['accuracy']:.2f}%")
    
    # Save detailed results if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test SA model on held-out speaker")
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument('--test_csv', type=str, required=True,
                        help="Path to test CSV file")
    parser.add_argument('--speaker_id', type=str, required=True,
                        help="Speaker ID being tested")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to save results JSON")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device for inference")
    args = parser.parse_args()
    
    # Load model
    asr_model = load_asr_model(args.checkpoint_dir, args.device)
    
    # Test on speaker
    results = test_model_on_speaker(
        asr_model, 
        args.test_csv, 
        args.speaker_id,
        args.output_file
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Speaker: {args.speaker_id}")
    print(f"WER: {results['wer']:.2f}%")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
