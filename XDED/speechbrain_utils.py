#!/usr/bin/env python3
"""
Utility functions for working with SpeechBrain models and data
for evaluation purposes in the SA model averaging experiments.
"""

import os
import torch
import json
import tqdm
import numpy as np
from pathlib import Path
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.metric_stats import wer_details_for_batch

def setup_inference_with_checkpoint(checkpoint_path, device="cuda"):
    """
    Load a SpeechBrain ASR model from a checkpoint path for inference.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model ready for inference
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Use the pretrained EncoderDecoderASR class from SpeechBrain
    # We need to customize this with the right hyperparams for SA models
    model = EncoderDecoderASR.from_hparams(
        source=checkpoint_path,  # May need to adjust this depending on how checkpoints are stored
        run_opts={"device": device},
        overrides={
            # Customizations for the SA model if needed
            "beam_size": 5,  # Reasonable beam size for inference
        }
    )
    
    return model

def load_testset(testset_path, speaker_id=None):
    """
    Load the test set for a specific speaker or all speakers.
    
    Args:
        testset_path: Path to test dataset JSON file
        speaker_id: Optional speaker ID to filter by
        
    Returns:
        DynamicItemDataset with test data
    """
    print(f"Loading test data from {testset_path}")
    
    # Load test data from JSON file
    with open(testset_path, 'r') as f:
        test_data = json.load(f)
    
    # Filter by speaker if specified
    if speaker_id:
        test_data = {k: v for k, v in test_data.items() if speaker_id in k}
        print(f"Filtered to {len(test_data)} utterances for speaker {speaker_id}")
    
    # Create a DynamicItemDataset (format used by SpeechBrain)
    dataset = DynamicItemDataset(test_data)
    
    # Define dynamic items (signal processing pipeline)
    dataset.add_dynamic_item(
        item_name="wav",
        func=lambda x: torch.tensor(x) if isinstance(x, (list, np.ndarray)) else x
    )
    
    # Set output keys
    dataset.set_output_keys(["id", "wav", "transcript"])
    
    return dataset

def compute_wer_for_testset(model, test_data, batch_size=8):
    """
    Compute Word Error Rate (WER) for a model on a test dataset.
    
    Args:
        model: SpeechBrain ASR model
        test_data: Test dataset
        batch_size: Batch size for inference
        
    Returns:
        Dict containing WER and other metrics
    """
    print("Computing WER for test set...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Track WER details
    wer_details = []
    
    # Process in batches
    for i in tqdm.tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        
        # Get waveforms and references
        wavs = [item["wav"] for item in batch]
        refs = [item["transcript"] for item in batch]
        
        # Inference
        with torch.no_grad():
            predictions = model.transcribe_batch(wavs)
        
        # Compute WER details
        batch_stats = wer_details_for_batch(predictions, refs)
        wer_details.extend(batch_stats)
    
    # Calculate overall WER
    total_words = sum(detail.num_ref_words for detail in wer_details)
    total_errors = sum(detail.num_errors for detail in wer_details)
    wer = (total_errors / total_words) * 100 if total_words > 0 else 100.0
    
    # Gather additional stats
    results = {
        "wer": wer,
        "num_utterances": len(test_data),
        "total_words": total_words,
        "total_errors": total_errors,
        "substitutions": sum(detail.num_substitutions for detail in wer_details),
        "deletions": sum(detail.num_deletions for detail in wer_details),
        "insertions": sum(detail.num_insertions for detail in wer_details),
    }
    
    print(f"WER: {wer:.2f}%")
    return results

def prepare_checkpoint_json(speakers, base_checkpoint_dir, output_path):
    """
    Create a JSON file mapping speaker IDs to checkpoint paths.
    
    Args:
        speakers: List of speaker IDs
        base_checkpoint_dir: Base directory containing speaker checkpoints
        output_path: Path to save the JSON file
    """
    checkpoint_paths = {}
    
    for speaker in speakers:
        # Format the path according to your directory structure
        speaker_dir = os.path.join(base_checkpoint_dir, f"val_uncommon_{speaker}")
        
        # Find the checkpoint file (adjust pattern as needed)
        checkpoint_files = list(Path(speaker_dir).glob("**/CKPT+*"))
        
        if checkpoint_files:
            # Use the most recent checkpoint (usually the last in alphabetical order)
            checkpoint_paths[speaker] = str(checkpoint_files[-1])
        else:
            print(f"Warning: No checkpoint found for speaker {speaker}")
    
    # Save the mapping to JSON
    with open(output_path, 'w') as f:
        json.dump(checkpoint_paths, f, indent=2)
    
    print(f"Checkpoint paths saved to {output_path}")
    print(f"Found checkpoints for {len(checkpoint_paths)} speakers")
    
    return checkpoint_paths