#!/usr/bin/env python3
"""
Evaluate pretrained NeMo Conformer-CTC on M08 test set as baseline.
"""

import os
import torch
import argparse
from tqdm import tqdm
import nemo.collections.asr as nemo_asr
from dassl.data.datasets.logit_ensemble import LogitEnsembleDataset


def _levenshtein_distance(ref_tokens, hyp_tokens):
    """Compute Levenshtein edit distance between two token lists.

    Parameters
    - ref_tokens: list[str]
    - hyp_tokens: list[str]

    Returns
    - int: edit distance
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    # dp[i][j] = distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[n][m]


def evaluate_pretrained_nemo(model_name, test_dataset, device='cuda', num_samples=30):
    """
    Evaluate pretrained NeMo model on test set.
    
    Args:
        model_name: NeMo model name (e.g., 'nvidia/stt_en_conformer_ctc_small')
        test_dataset: LogitEnsembleDataset for test
        device: Device to run on
        num_samples: Number of samples to show detailed predictions
    
    Returns:
        Dictionary with evaluation results
    """
    # Load pretrained NeMo model
    print(f"\nLoading pretrained NeMo model: {model_name}")
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Sample rate: {model.cfg.preprocessor.sample_rate if hasattr(model, 'cfg') else 16000}")
    
    # Evaluation loop
    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    print(f"Showing detailed predictions for first {num_samples} samples:\n")
    
    all_predictions = []
    all_targets = []
    all_utt_ids = []
    wra_total = 0
    wra_correct = 0
    # WER accumulators (word-level Levenshtein)
    wer_edits_total = 0
    wer_ref_words_total = 0

    def _normalize_word(s: str) -> str:
        return (s or "").strip().lower()
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            # Get batch
            batch = test_dataset[idx]
            
            # Prepare inputs - NeMo expects [batch, time] waveform at 16kHz
            audio = batch['audio'].unsqueeze(0).to(device)  # [1, time]
            
            # NeMo transcribe expects list of numpy arrays or file paths
            # We'll use the model's forward + decode instead
            audio_signal = audio
            audio_signal_len = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)
            
            # Run NeMo model forward pass
            log_probs, encoded_len, predictions = model.forward(
                input_signal=audio_signal,
                input_signal_length=audio_signal_len
            )
            
            # Decode using NeMo's transcribe method (simpler)
            # Convert tensor back for transcribe
            hypotheses = model.decoding.ctc_decoder_predictions_tensor(
                predictions,
                encoded_len,
                return_hypotheses=False,
            )
            
            # Extract text from hypothesis
            if hypotheses and len(hypotheses) > 0:
                hyp = hypotheses[0]
                # Handle both string and Hypothesis object
                if isinstance(hyp, str):
                    predicted_text = hyp
                else:
                    predicted_text = hyp.text if hasattr(hyp, 'text') else str(hyp)
            else:
                predicted_text = ""
            target_text = batch['target_text']
            
            # Store results
            all_predictions.append(predicted_text)
            all_targets.append(target_text)
            all_utt_ids.append(batch['utterance_id'])
            
            # Compute WRA on first word
            if isinstance(target_text, str) and target_text.strip() != "":
                ref_first = _normalize_word(target_text)
                hyp_words = predicted_text.strip().split()
                hyp_first = _normalize_word(hyp_words[0] if hyp_words else "")
                wra_total += 1
                if ref_first == hyp_first:
                    wra_correct += 1

            # Compute sentence-level WER (word-level Levenshtein on full utterance)
            ref_tokens = (target_text or "").strip().lower().split()
            hyp_tokens = (predicted_text or "").strip().lower().split()
            if ref_tokens:
                dist = _levenshtein_distance(ref_tokens, hyp_tokens)
                wer_edits_total += dist
                wer_ref_words_total += len(ref_tokens)
            
            # Show detailed output for first N samples
            if idx < num_samples:
                print(f"\n{'='*80}")
                print(f"Sample {idx+1}/{num_samples}")
                print(f"Utterance ID: {batch['utterance_id']}")
                print(f"Speaker: {batch['speaker']}")
                print(f"Target: {target_text}")
                print(f"Predicted: {predicted_text}")
                # Only print match if a non-empty reference exists
                if isinstance(target_text, str) and target_text.strip() != "":
                    print(f"Match: {'✓' if ref_first == hyp_first else '✗'}")
    
    # Summary statistics
    num_samples_eval = len(all_predictions)
    empty_pred_frac = sum(1 for pred in all_predictions if not pred.strip()) / max(1, num_samples_eval)
    unique_preds = len(set(all_predictions))

    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples evaluated: {num_samples_eval}")
    print(f"Empty predictions: {empty_pred_frac*100:.1f}%")
    print(f"Unique predictions: {unique_preds}/{num_samples_eval}")
    
    avg_hyp_words = sum(len((txt or "").strip().split()) for txt in all_predictions) / max(1, num_samples_eval)
    print(f"Average decoded words per utt: {avg_hyp_words:.2f}")

    # WRA for single-word evaluation
    wra_value = (wra_correct / max(1, wra_total)) if wra_total > 0 else None
    if wra_value is not None:
        print(f"WRA (first-word match, case-insensitive): {wra_value*100:.2f}%  ({wra_correct}/{wra_total})")

    # WER: total edit distance over total reference words
    wer_value = (wer_edits_total / max(1, wer_ref_words_total)) if wer_ref_words_total > 0 else None
    if wer_value is not None:
        print(f"WER (word-level Levenshtein): {wer_value*100:.2f}%  (edits={wer_edits_total}, ref_words={wer_ref_words_total})")
    
    # Save detailed results
    results = {
        'model': model_name,
        'num_samples': num_samples_eval,
        'wra': {
            'value': wra_value,
            'correct': wra_correct,
            'total': wra_total,
        },
        'wer': {
            'value': wer_value,
            'edits_total': wer_edits_total,
            'ref_words_total': wer_ref_words_total,
        },
        'predictions': [
            {
                'utterance_id': utt_id,
                'target': target,
                'prediction': pred,
            }
            for utt_id, target, pred in zip(all_utt_ids, all_targets, all_predictions)
        ]
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate pretrained NeMo model')
    parser.add_argument('--model_name', type=str, 
                       default='nvidia/stt_en_conformer_ctc_small',
                       help='NeMo model name')
    parser.add_argument('--held_out', type=str, default='M08',
                       help='Held-out speaker to evaluate on')
    parser.add_argument('--matching_mode', type=str, default='partial',
                       choices=['strict', 'partial', 'all'],
                       help='Matching mode for ensemble loading')
    parser.add_argument('--min_teachers', type=int, default=10,
                       help='Minimum teachers for partial mode')
    parser.add_argument('--num_samples', type=int, default=30,
                       help='Number of detailed samples to show')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--output', type=str, default='eval_nemo_pretrained_M08.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Configuration
    csv_dir = "/home/zsim710/partitions/uaspeech/by_speakers"
    
    print("="*80)
    print("PRETRAINED NEMO MODEL EVALUATION (BASELINE)")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Held-out speaker: {args.held_out}")
    print(f"Device: {args.device}")
    
    # Create test dataset for held-out speaker
    # We'll use LogitEnsembleDataset but only use audio and targets
    print(f"\nCreating test dataset for {args.held_out}...")
    logit_root_dir = "/home/zsim710/XDED/speechbrain/exp_results/logit_extraction"
    
    test_dataset = LogitEnsembleDataset(
        logit_root_dir=logit_root_dir,
        held_out_speaker=args.held_out,
        split='test',
        matching_mode=args.matching_mode,
        min_teachers=args.min_teachers,
        exclude_speakers=['M01'],
        csv_dir=csv_dir
    )
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Evaluate model
    import json
    results = evaluate_pretrained_nemo(
        model_name=args.model_name,
        test_dataset=test_dataset,
        device=args.device,
        num_samples=args.num_samples
    )
    
    # Save results
    output_path = args.output
    print(f"\nSaving detailed results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
