#!/usr/bin/env python3
"""
Extract logits from Speaker-Adaptive model checkpoints.
This script extracts both CTC encoder logits and teacher-forced decoder logits
from each SA model when tested on its own speaker dataset.

Focus is on teacher-forced decoder logits for knowledge distillation.
"""

import os
import sys
import torch
import logging
import numpy as np
import argparse
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
import json

# Import your existing classes
from train import ASR, dataio_prepare

# Disable wandb and set GPU
os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Verify GPU setup
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs visible: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")  # Will be GPU 2 due to CUDA_VISIBLE_DEVICES
else:
    print("CUDA not available!")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SA Model checkpoints
SA_MODEL_CHECKPOINTS = {
    "F02": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F02_E0D2/7775/save/CKPT+2024-07-11+17-09-55+00",
    "F03": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F03_E0D3/7775/save/CKPT+2024-07-11+20-38-16+00",
    "F04": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F04_E0D2/7775/save/CKPT+2024-07-11+17-39-01+00",
    "F05": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F05_E0D2/7775/save/CKPT+2024-07-11+17-53-21+00",
    "M01": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M01_E0D4/7775/save/CKPT+2024-07-12+18-22-00+00",
    "M04": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M04_E0D2/7775/save/CKPT+2024-07-11+18-19-53+00",
    "M05": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M05_E1D0/7775/save/CKPT+2024-07-12+11-58-04+00",
    "M07": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M07_E0D0/7775/save/CKPT+2024-07-11+12-18-47+00",
    "M08": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00",
    "M09": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M09_E0D0/7775/save/CKPT+2024-07-11+12-42-43+00",
    "M10": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M10_E1D0/7775/save/CKPT+2024-07-12+12-51-27+00",
    "M11": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M11_E0D1/7775/save/CKPT+2024-07-11+16-21-04+00",
    "M12": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M12_E0D4/7775/save/CKPT+2024-07-12+20-01-22+00",
    "M14": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M14_E0D0/7775/save/CKPT+2024-07-11+13-35-20+00",
    "M16": "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M16_E0D2/7775/save/CKPT+2024-07-11+20-10-30+00",
}

class LogitExtractorASR(ASR):
    """Modified ASR class specifically for logit extraction."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit_storage = {
            'decoder_logits': [],       # Teacher-forced decoder logits [B, U, V] 
            'token_lens': [],           # Token lengths for decoder alignment
            'ctc_logits': [],           # CTC encoder logits [B, T, V]
            'frame_lens': [],           # Frame lengths (T) for CTC alignment
            'utterance_ids': [],        # Utterance IDs
            'target_tokens': [],        # Ground truth tokens
            'target_words': [],         # Ground truth words
        }
        self.tokenizer = None  # Initialize tokenizer attribute
    
    def compute_forward(self, batch, stage):
        """Modified forward pass to extract and store logits."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        # Compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # Forward modules
        src = self.modules.CNN(feats)
        
        # Get encoder output and decoder predictions (teacher-forced)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )

        # Extract logits BEFORE softmax (these are the raw logits we want for KD)
        # CTC logits: [B, T, V] - frame-wise acoustic distributions
        ctc_logits = self.modules.ctc_lin(enc_out)
        
        # Decoder logits: [B, U, V] - teacher-forced token-wise distributions  
        decoder_logits = self.modules.seq_lin(pred)

        # Store logits and metadata during test stage
        if stage == sb.Stage.TEST:
            # Debug: print stage and shapes
            logger.debug(f"Stage: {stage}, storing decoder logits with shape: {decoder_logits.shape}")
            
            # Store raw decoder logits (pre-softmax) for knowledge distillation
            self.logit_storage['decoder_logits'].append(decoder_logits.detach().cpu())
            # Store raw CTC logits (pre-softmax)
            self.logit_storage['ctc_logits'].append(ctc_logits.detach().cpu())
            
            # Store lengths for proper alignment
            self.logit_storage['token_lens'].append(tokens_eos_lens.detach().cpu())
            # For CTC, store the time dimension T for each sample in batch
            # Note: we use actual logits shape to avoid assumptions about subsampling
            self.logit_storage['frame_lens'].append(torch.tensor([ctc_logits.shape[1]], dtype=torch.long))
            
            # Store IDs and targets
            self.logit_storage['utterance_ids'].extend(batch.id)
            self.logit_storage['target_tokens'].append(tokens_eos.detach().cpu())
            self.logit_storage['target_words'].extend(batch.wrd)
            
            # Clear GPU cache periodically to prevent memory buildup
            if len(self.logit_storage['utterance_ids']) % 100 == 0:
                torch.cuda.empty_cache()
        else:
            logger.debug(f"Not TEST stage, current stage: {stage}")

        # Continue with normal processing for metrics
        p_ctc = self.hparams.log_softmax(ctc_logits)
        p_seq = self.hparams.log_softmax(decoder_logits)

        # Generate hypotheses for evaluation
        hyps = None
        is_test_search = stage == sb.Stage.TEST
        
        if is_test_search:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), wav_lens)
        else:
            # Greedy decoding for validation
            hyps = []
            for seq in p_seq.argmax(-1):
                eos_index_array = (seq == 2).nonzero(as_tuple=True)[0]
                if len(eos_index_array):
                    eos_index = eos_index_array[0]
                    hyp = seq[:eos_index].tolist()
                else:
                    hyp = seq.tolist()
                hyps.append(hyp)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Modified compute_objectives to properly decode tokens."""
        p_ctc, p_seq, wav_lens, hyps = predictions
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        # Compute CTC loss
        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        )

        # Compute seq2seq loss
        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, tokens_eos_lens
        )

        # Total loss
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            
            # Decode token terms to words using the tokenizer properly
            # Use the tokenizer from hparams which has the decode_ids method
            predicted_words = [
                self.hparams.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
            self.wra_metric.append(predicted_words, target_words)
            self.inference_results.extend(zip(ids, target_words, predicted_words))
        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        """Override on_stage_end to skip file writing that requires test_wer_file."""
        if stage != sb.Stage.TRAIN:
            stage_stats = self.wer_metric.summarize("error_rate")
            
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"]
            )
        elif stage == sb.Stage.TEST:
            # If stage_stats is just a float (WER), wrap it in a dict
            if isinstance(stage_stats, (float, int)):
                stage_stats = {"WER": stage_stats}
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            # Skip the file writing part that requires test_wer_file for logit extraction

    def save_extracted_logits(self, output_dir, model_name):
        """Save the extracted logits to files"""
        logger.info(f"Saving logits for {model_name}")
        
        # Additional debug: Check if logit_storage is actually a dictionary
        if not hasattr(self.logit_storage, 'items'):
            logger.error(f"ERROR: logit_storage does not have items() method! Type: {type(self.logit_storage)}, Value: {self.logit_storage}")
            return False
        
        try:
            # Ensure output directory exists
            model_output_dir = f"{output_dir}/{model_name}"
            import os
            os.makedirs(model_output_dir, exist_ok=True)
            logger.info(f"Created/using output directory: {model_output_dir}")

            # Save decoder logits (if available) without overwriting existing files
            try:
                dec_out_path = f"{model_output_dir}/{model_name}_decoder_logits.pt"
                if os.path.exists(dec_out_path):
                    logger.info(f"Decoder logits already exist at {dec_out_path}, skipping overwrite.")
                elif self.logit_storage['decoder_logits']:
                    logger.info(f"Processing {len(self.logit_storage['decoder_logits'])} decoder logits")
                    # Check if all decoder logits have the same sequence length
                    seq_lengths = [x.shape[1] for x in self.logit_storage['decoder_logits']]
                    logger.info(f"Decoder sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}")
                    
                    if len(set(seq_lengths)) == 1:
                        # All same length, can concatenate directly
                        decoder_logits = torch.cat(self.logit_storage['decoder_logits'], dim=0)
                        logger.info(f"Decoder logits concatenated: {decoder_logits.shape}")
                        torch.save(decoder_logits, dec_out_path)
                    else:
                        # Variable lengths, save as list of tensors
                        logger.info(f"Variable decoder lengths detected, saving as list")
                        decoder_logits = [x.squeeze(0) for x in self.logit_storage['decoder_logits']]  # Remove batch dim
                        torch.save(decoder_logits, dec_out_path)
                else:
                    logger.info("No decoder logits collected this run; nothing to save for decoder.")
                    
            except Exception as e:
                logger.error(f"Failed to save decoder logits: {e}")
                logger.error(f"Decoder logit types: {[type(x) for x in self.logit_storage.get('decoder_logits', [])[:3]]}")
                # Continue even if decoder saving fails; we still want to try CTC
            
            # Save CTC logits (new)
            try:
                ctc_out_path = f"{model_output_dir}/{model_name}_ctc_logits.pt"
                if os.path.exists(ctc_out_path):
                    logger.info(f"CTC logits already exist at {ctc_out_path}, skipping overwrite.")
                elif self.logit_storage['ctc_logits']:
                    logger.info(f"Processing {len(self.logit_storage['ctc_logits'])} CTC logits")
                    # Decoder may occasionally be fixed-length; CTC almost always variable, save list
                    # Remove batch dim if B==1
                    ctc_logits = [x.squeeze(0) for x in self.logit_storage['ctc_logits']]
                    torch.save(ctc_logits, ctc_out_path)
                else:
                    logger.warning("No CTC logits collected; nothing to save for CTC.")
            except Exception as e:
                logger.error(f"Failed to save CTC logits: {e}")
                return False

            # Save metadata
            vocab_size_dec = self.logit_storage['decoder_logits'][0].shape[-1] if self.logit_storage['decoder_logits'] else None
            vocab_size_ctc = self.logit_storage['ctc_logits'][0].shape[-1] if self.logit_storage['ctc_logits'] else None
                
            # If a metadata file already exists, load and update it; else create new
            metadata_path = f"{model_output_dir}/{model_name}_metadata.json"
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    logger.info("Loaded existing metadata.json; updating with CTC fields if needed.")
                except Exception:
                    metadata = {}
            else:
                metadata = {}

            # Merge/augment metadata
            metadata.update({
                'model_name': model_name,
                'num_utterances': len(self.logit_storage['utterance_ids']) or metadata.get('num_utterances'),
                'vocab_size_decoder': vocab_size_dec or metadata.get('vocab_size_decoder'),
                'vocab_size_ctc': vocab_size_ctc or metadata.get('vocab_size_ctc'),
                'utterance_ids': self.logit_storage['utterance_ids'] or metadata.get('utterance_ids', []),
                'token_lens': [item.tolist() if torch.is_tensor(item) else item for item in self.logit_storage['token_lens']] or metadata.get('token_lens', []),
                'ctc_frame_lens': [int(item.squeeze().item()) if torch.is_tensor(item) else int(item) for item in self.logit_storage['frame_lens']] or metadata.get('ctc_frame_lens', []),
                'target_words': self.logit_storage['target_words'] or metadata.get('target_words', [])
            })
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully saved logits for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in save_extracted_logits for {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def extract_logits_for_sa_model(sa_model, hparams_file, output_base_dir):
    """Extract logits for a specific SA model on its own test dataset."""
    
    logger.info(f"Extracting logits for SA model: {sa_model}")
    
    # Load and modify hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, {})
    
    # Update for current SA model (test on same speaker)
    checkpoint_path = SA_MODEL_CHECKPOINTS[sa_model] 
    hparams['sa_model'] = sa_model
    hparams['speaker'] = sa_model  # Test on same speaker's data
    hparams['load_ckpt'] = checkpoint_path
    hparams['test_csv'] = [f'/home/zsim710/partitions/uaspeech/by_speakers/{sa_model}.csv']
    
    # Update checkpointer to point to the correct SA model checkpoint directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    hparams['checkpointer'].checkpoints_dir = Path(checkpoint_dir)
    
    # Create output directory for this SA model
    output_dir = os.path.join(output_base_dir, sa_model)
    
    # Prepare data
    try:
        _, _, test_datasets, tokenizer, _, _ = dataio_prepare(hparams)
        logger.info(f"Loaded test dataset for {sa_model}")
    except Exception as e:
        logger.error(f"Failed to prepare data for {sa_model}: {e}")
        return None

    # Load pretrained models (tokenizer and LM) - this is critical!
    try:
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()
        logger.info(f"Loaded pretrained tokenizer for {sa_model}")
    except Exception as e:
        logger.error(f"Failed to load pretrained tokenizer for {sa_model}: {e}")
        return None
    
    # Initialize model with logit extraction capabilities
    asr_brain = LogitExtractorASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts={'device': 'cuda:0'},  # This will be GPU 2 due to CUDA_VISIBLE_DEVICES=2
        checkpointer=hparams["checkpointer"],
    )
    # Set the tokenizer properly
    asr_brain.tokenizer = tokenizer
    asr_brain.hparams.tokenizer = tokenizer
    
    # Clear any previous logit storage
    asr_brain.logit_storage = {
        'decoder_logits': [],       
        'token_lens': [],           
        'ctc_logits': [],
        'frame_lens': [],
        'utterance_ids': [],        
        'target_tokens': [],        
        'target_words': [],         
    }
    
    # Load checkpoint
    try:
        ckpt = asr_brain.checkpointer.find_checkpoint(
            max_key=hparams.get("max_metric_key"), 
            min_key=hparams.get("min_metric_key")
        )
        asr_brain.checkpointer.load_checkpoint(ckpt)
        asr_brain.modules.eval()
        logger.info(f"Loaded checkpoint for {sa_model}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint for {sa_model}: {e}")
        return None
    
    # Extract logits by running evaluation on test set
    try:
        test_dataset_name = list(test_datasets.keys())[0]  # Should be the sa_model name
        test_dataset = test_datasets[test_dataset_name]
        
        logger.info(f"Starting logit extraction for {sa_model} on {len(test_dataset)} utterances...")
        
        # Run evaluation to trigger logit extraction
        asr_brain.evaluate(
            test_dataset,
            max_key="ACC",
            test_loader_kwargs={'batch_size': 1, 'shuffle': False}  # Use batch_size=1 for simplicity
        )
        
        # Save extracted logits (will append CTC alongside existing decoder without overwriting)
        logits_data = asr_brain.save_extracted_logits(output_dir, sa_model)
        
        # Clear GPU memory after processing each model
        torch.cuda.empty_cache()
        
        logger.info(f"✅ Successfully extracted logits for {sa_model}")
        return logits_data
        
    except Exception as e:
        logger.error(f"Failed during logit extraction for {sa_model}: {e}")
        return None


def main():
    """Main function to extract logits from SA models."""
    
    # Parse command line arguments with defaults
    parser = argparse.ArgumentParser(description='Extract logits from Speaker-Adaptive models')
    parser.add_argument('hparams_file', nargs='?', 
                       default='hparams/exp/uaspeech/ua_SA_val_uncommon_WRA.yaml',
                       help='Path to hyperparameters YAML file')
    parser.add_argument('--data_folder', 
                       default='/home/zsim710/XDED/speechbrain/datasets/UASpeech',
                       help='Path to UASpeech dataset')
    parser.add_argument('--output_folder', 
                       default='/home/zsim710/XDED/speechbrain/exp_results/logit_extraction',
                       help='Output directory for logit files')
    parser.add_argument('--models', nargs='+', help='List of SA models to process (e.g., F03 F04)', 
                   default=list(SA_MODEL_CHECKPOINTS.keys()))  # Process ALL models by default
    
    args = parser.parse_args()
    
    # Configuration
    hparams_file = args.hparams_file
    output_base_dir = args.output_folder
    
    # Filter models if specified
    models_to_process = [model for model in args.models if model in SA_MODEL_CHECKPOINTS]
    if not models_to_process:
        logger.error(f"No valid models specified. Available models: {list(SA_MODEL_CHECKPOINTS.keys())}")
        sys.exit(1)
    
    logger.info("Starting SA Model Logit Extraction...")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"Number of SA models: {len(models_to_process)}")
    
    # Check prerequisites
    if not os.path.exists(hparams_file):
        logger.error(f"YAML file not found: {hparams_file}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Track results
    successful_extractions = []
    failed_extractions = []
    
    # Extract logits for each SA model
    for sa_model in models_to_process:
        try:
            # Check if CTC logits already exist; if yes, skip. Otherwise process.
            model_output_dir = os.path.join(output_base_dir, sa_model, sa_model)
            ctc_file = os.path.join(model_output_dir, f"{sa_model}_ctc_logits.pt")
            if os.path.exists(ctc_file):
                logger.info(f"✅ {sa_model} CTC logits already exist, skipping...")
                successful_extractions.append(sa_model)
                continue
                
            logits_data = extract_logits_for_sa_model(sa_model, hparams_file, output_base_dir)
            if logits_data is not None:
                successful_extractions.append(sa_model)
            else:
                failed_extractions.append(sa_model)
        except Exception as e:
            logger.error(f"Exception during extraction for {sa_model}: {e}")
            failed_extractions.append(sa_model)
            # Clear GPU memory after failure
            torch.cuda.empty_cache()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("LOGIT EXTRACTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total SA models: {len(SA_MODEL_CHECKPOINTS)}")
    logger.info(f"Successful extractions: {len(successful_extractions)}")
    logger.info(f"Failed extractions: {len(failed_extractions)}")
    
    if successful_extractions:
        logger.info(f"\n✅ Successful extractions:")
        for model in successful_extractions:
            logger.info(f"  - {model}")
    
    if failed_extractions:
        logger.info(f"\n❌ Failed extractions:")
        for model in failed_extractions:
            logger.info(f"  - {model}")
    
    # Save summary
    summary = {
        'successful_extractions': successful_extractions,
        'failed_extractions': failed_extractions,
        'total_models': len(SA_MODEL_CHECKPOINTS),
        'output_directory': output_base_dir,
    }
    
    summary_file = os.path.join(output_base_dir, "extraction_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nExtraction complete! Summary saved to: {summary_file}")
    logger.info(f"Logits saved in: {output_base_dir}")


if __name__ == "__main__":
    main()