#!/usr/bin/env python3
"""
Precompute utterance-level entropies for curriculum learning.
Speaker WERs are already known and hardcoded.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths
LOGIT_ROOT = "/home/zsim710/XDED/speechbrain/exp_results/logit_extraction"
OUTPUT_FILE = "/home/zsim710/XDED/XDED/curriculum_difficulty_scores.json"

# Known speaker WERs (from SA model evaluation on their own speakers)
SPEAKER_WERS = {
    'F05': 2.05,
    'M07': 6.9,
    'M10': 3.85,
    'M14': 9.8,
    'M09': 5.5,
    'F04': 17.6,
    'M11': 17.15,
    'M05': 13.6,
    'M16': 14.65,
    'F02': 16.85,
    'M01': 42.3,
    'M12': 47.0,
    'F03': 44.15,
    'M04': 86.4,
}

# Intelligibility band mapping (from UA-Speech paper)
INTELLIGIBILITY_BANDS = {
    'M01': 'VERY_LOW',
    'M04': 'VERY_LOW',
    'M05': 'MID',
    'M07': 'LOW',
    'M08': 'HIGH',
    'M09': 'HIGH',
    'M10': 'HIGH',
    'M11': 'MID',
    'M12': 'VERY_LOW',
    'M14': 'HIGH',
    'M16': 'LOW',
    'F02': 'LOW',
    'F03': 'VERY_LOW',
    'F04': 'MID',
    'F05': 'HIGH',
}


def compute_entropy(decoder_logits):
    """
    Compute entropy of decoder logits.
    
    Args:
        decoder_logits: [L, V] - decoder logits
    
    Returns:
        Scalar entropy value (higher = more uncertain)
    """
    # Compute probabilities and entropy
    probs = F.softmax(decoder_logits, dim=-1)  # [L, V]
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)  # [L]
    
    # Return mean entropy over sequence
    return entropy.mean().item()


def precompute_utterance_entropies(speakers, logit_root):
    """Compute entropy for all utterances across all speakers."""
    utterance_scores = {}
    
    for speaker in tqdm(speakers, desc="Computing entropies"):
        # Path structure: logit_root/speaker/speaker/speaker_decoder_logits.pt
        logit_file = Path(logit_root) / speaker / speaker / f"{speaker}_decoder_logits.pt"
        metadata_file = Path(logit_root) / speaker / speaker / f"{speaker}_metadata.json"
        
        if not logit_file.exists():
            print(f"⚠️  Warning: Logit file not found: {logit_file}")
            continue
        
        if not metadata_file.exists():
            print(f"⚠️  Warning: Metadata file not found: {metadata_file}")
            continue
        
        try:
            # Load decoder logits (list of tensors, one per utterance)
            logits_list = torch.load(logit_file, map_location='cpu')
            
            # Load metadata to get utterance IDs
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            utterance_ids = metadata.get('utterance_ids', metadata.get('utterances', []))
            
            if len(logits_list) != len(utterance_ids):
                print(f"⚠️  Warning: Mismatch for {speaker}: {len(logits_list)} logits vs {len(utterance_ids)} IDs")
            
            print(f"   Processing {speaker}: {len(logits_list)} utterances")
            
            # Process each utterance
            for idx, logits in enumerate(logits_list):
                utt_id = utterance_ids[idx] if idx < len(utterance_ids) else f"{speaker}_{idx}"
                
                # Remove speaker prefix to get core ID (e.g., "M08_B3_D3_M2" -> "B3_D3_M2")
                if '_' in utt_id:
                    core_id = utt_id.split('_', 1)[1]
                else:
                    core_id = utt_id
                
                # Compute entropy
                entropy = compute_entropy(logits)
                
                utterance_scores[core_id] = {
                    'speaker': speaker,
                    'full_id': utt_id,
                    'core_id': core_id,
                    'entropy': entropy
                }
        except Exception as e:
            print(f"⚠️  Failed to process {speaker}: {e}")
    
    print(f"\n✅ Computed entropies for {len(utterance_scores)} utterances")
    return utterance_scores


def combine_difficulty_scores(speaker_wers, utterance_scores, alpha=0.6, beta=0.4):
    """
    Combine speaker WER and utterance entropy into final difficulty score.
    
    Args:
        speaker_wers: Dict mapping speaker -> WER percentage
        utterance_scores: Dict with utterance entropy data
        alpha: Weight for speaker-level difficulty (0-1)
        beta: Weight for utterance-level difficulty (0-1)
    
    Returns:
        Dict mapping utterance_id -> combined_difficulty_score [0, 1]
    """
    # Normalize speaker WERs to [0, 1]
    wer_values = list(speaker_wers.values())
    min_wer, max_wer = min(wer_values), max(wer_values)
    normalized_wers = {
        spk: (wer - min_wer) / (max_wer - min_wer + 1e-9)
        for spk, wer in speaker_wers.items()
    }
    
    # Normalize entropies to [0, 1]
    entropies = [u['entropy'] for u in utterance_scores.values()]
    min_ent, max_ent = min(entropies), max(entropies)
    
    combined = {}
    for utt_id, utt_data in utterance_scores.items():
        speaker = utt_data['speaker']
        
        # Speaker difficulty (from normalized WER)
        speaker_difficulty = normalized_wers.get(speaker, 0.5)
        
        # Utterance difficulty (normalized entropy)
        utterance_difficulty = (utt_data['entropy'] - min_ent) / (max_ent - min_ent + 1e-9)
        
        # Combined score
        combined_score = alpha * speaker_difficulty + beta * utterance_difficulty
        
        combined[utt_id] = {
            'speaker': speaker,
            'speaker_wer': speaker_wers.get(speaker, 0.0),
            'speaker_difficulty': speaker_difficulty,
            'utterance_difficulty': utterance_difficulty,
            'combined_difficulty': combined_score,
            'entropy': utt_data['entropy'],
            'intelligibility_band': INTELLIGIBILITY_BANDS.get(speaker, 'UNKNOWN')
        }
    
    return combined


def main():
    print("="*80)
    print("PRECOMPUTING CURRICULUM DIFFICULTY SCORES")
    print("="*80)
    
    # 1. Display speaker WERs
    print("\n📊 Known Speaker WERs (lower = easier):")
    sorted_speakers = sorted(SPEAKER_WERS.items(), key=lambda x: x[1])
    for speaker, wer in sorted_speakers:
        band = INTELLIGIBILITY_BANDS.get(speaker, 'UNKNOWN')
        print(f"   {speaker}: {wer:5.2f}% WER  (Band: {band})")
    
    # 2. Compute utterance entropies
    speakers = list(SPEAKER_WERS.keys())
    utterance_scores = precompute_utterance_entropies(speakers, LOGIT_ROOT)
    
    if not utterance_scores:
        print("\n❌ ERROR: No utterance scores computed. Check logit paths.")
        return
    
    # 3. Combine into final difficulty scores
    combined_scores = combine_difficulty_scores(SPEAKER_WERS, utterance_scores, alpha=0.6, beta=0.4)
    
    # 4. Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(combined_scores, f, indent=2)
    
    print(f"\n✅ Saved difficulty scores to: {OUTPUT_FILE}")
    
    # 5. Print statistics
    difficulties = [s['combined_difficulty'] for s in combined_scores.values()]
    print(f"\n📊 Difficulty Statistics:")
    print(f"   Total utterances: {len(difficulties)}")
    print(f"   Min: {min(difficulties):.3f}")
    print(f"   Max: {max(difficulties):.3f}")
    print(f"   Mean: {np.mean(difficulties):.3f}")
    print(f"   Median: {np.median(difficulties):.3f}")
    
    # Show examples
    print(f"\n📋 Example Difficulties:")
    sorted_utts = sorted(combined_scores.items(), key=lambda x: x[1]['combined_difficulty'])
    print("\n   Easiest 5:")
    for utt_id, scores in sorted_utts[:5]:
        print(f"      {utt_id}: {scores['combined_difficulty']:.3f} "
              f"(Speaker: {scores['speaker']}, WER: {scores['speaker_wer']:.1f}%, Entropy: {scores['entropy']:.2f})")
    print("\n   Hardest 5:")
    for utt_id, scores in sorted_utts[-5:]:
        print(f"      {utt_id}: {scores['combined_difficulty']:.3f} "
              f"(Speaker: {scores['speaker']}, WER: {scores['speaker_wer']:.1f}%, Entropy: {scores['entropy']:.2f})")
    
    # 6. Print distribution by speaker
    print(f"\n📊 Difficulty Distribution by Speaker:")
    speaker_difficulties = {}
    for utt_id, scores in combined_scores.items():
        spk = scores['speaker']
        if spk not in speaker_difficulties:
            speaker_difficulties[spk] = []
        speaker_difficulties[spk].append(scores['combined_difficulty'])
    
    for speaker in sorted(speaker_difficulties.keys(), key=lambda s: SPEAKER_WERS.get(s, 0)):
        diffs = speaker_difficulties[speaker]
        wer = SPEAKER_WERS.get(speaker, 0)
        print(f"   {speaker} (WER: {wer:5.2f}%): {len(diffs):4d} utts, "
              f"avg difficulty: {np.mean(diffs):.3f}")


if __name__ == '__main__':
    main()
