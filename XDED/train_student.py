#!/usr/bin/env python3
"""
Training script for Student Conformer via Knowledge Distillation

Train a lightweight student Conformer to match ensemble of 14 SA teacher models.
Uses KL divergence loss with temperature scaling for logit-level distillation.
"""

import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import json
from pathlib import Path

# Add XDED to path
sys.path.insert(0, '/home/zsim710/XDED/XDED')

from dassl.data.datasets.logit_ensemble import LogitEnsembleDataset, collate_logits
from models.student_conformer import StudentConformer
try:
    from models.nemo_hybrid_student import NeMoHybridStudent  # Optional dependency
    _HAS_NEMO = True
except Exception:
    NeMoHybridStudent = None
    _HAS_NEMO = False


def get_curriculum_competence(epoch, total_epochs, schedule='sqrt'):
    """
    Calculate curriculum learning competence based on training progress.
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        schedule: Curriculum schedule type
            - 'none': No curriculum, always use all data (competence=1.0)
            - 'linear': Linear growth from 0 to 1
            - 'sqrt': Square root schedule (faster initial growth)
            - 'step': Step-wise increase at 25%, 50%, 75% progress
    
    Returns:
        competence: Float in [0, 1] indicating portion of data to use
    """
    if schedule == 'none':
        return 1.0
    
    progress = epoch / max(1, total_epochs)
    
    if schedule == 'linear':
        return progress
    elif schedule == 'sqrt':
        return math.sqrt(progress)
    elif schedule == 'step':
        if progress < 0.25:
            return 0.3
        elif progress < 0.5:
            return 0.6
        elif progress < 0.75:
            return 0.8
        else:
            return 1.0
    else:
        raise ValueError(f"Unknown curriculum schedule: {schedule}")



def aggregate_teacher_distributions(
    teacher_logits: torch.Tensor,
    lengths_list,
    num_teachers: torch.Tensor,
    temperature: float = 2.0,
    teacher_agg: str = "prob_mean",
    eps: float = 1e-8,
):
    """Aggregate teacher distributions across teachers per time step.

    Args:
        teacher_logits: [B, T, Lt, V]
        lengths_list: list of len B, each a 1D tensor with per-teacher valid lengths
        num_teachers: [B]
        temperature: temperature for softmax/log-softmax
        teacher_agg: one of {'prob_mean','logprob_mean','logit_mean'}

    Returns:
        teacher_avg: [B, Lt, V] averaged distribution per time step
        teacher_mask: [B, T, Lt] boolean mask of valid teacher frames
        teacher_count: [B, Lt] count of valid teachers per frame
    """
    device = teacher_logits.device
    B, Tch, Lt, V = teacher_logits.shape

    # Temperature-scaled probabilities/log-probabilities
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)  # [B, T, Lt, V]
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)  # [B, T, Lt, V]

    # Build mask of valid teacher frames: [B, T, Lt]
    teacher_mask = torch.zeros((B, Tch, Lt), dtype=torch.bool, device=device)
    for b in range(B):
        nt = int(num_teachers[b].item()) if isinstance(num_teachers, torch.Tensor) else int(num_teachers[b])
        lens_b = lengths_list[b]
        for t in range(nt):
            l = int(lens_b[t].item())
            if l > 0:
                teacher_mask[b, t, : min(l, Lt)] = True

    # Count valid teachers per frame: [B, Lt]
    teacher_count = teacher_mask.sum(dim=1)  # [B, Lt]
    teacher_count_clamped = teacher_count.clamp(min=1).unsqueeze(-1)  # [B, Lt, 1]

    # Aggregate teacher distributions across teachers (dimension 1)
    if teacher_agg == "prob_mean":
        tp = teacher_probs * teacher_mask.unsqueeze(-1).to(teacher_probs.dtype)
        teacher_avg = tp.sum(dim=1) / teacher_count_clamped
    elif teacher_agg == "logprob_mean":
        tlp = teacher_log_probs * teacher_mask.unsqueeze(-1).to(teacher_log_probs.dtype)
        logp_avg = tlp.sum(dim=1) / teacher_count_clamped  # [B, Lt, V]
        teacher_avg = torch.exp(logp_avg)
        teacher_avg = teacher_avg / teacher_avg.sum(dim=-1, keepdim=True).clamp_min(eps)
    elif teacher_agg == "logit_mean":
        tl = teacher_logits * teacher_mask.unsqueeze(-1).to(teacher_logits.dtype)
        logit_avg = tl.sum(dim=1) / teacher_count_clamped  # [B, Lt, V]
        teacher_avg = F.softmax(logit_avg / temperature, dim=-1)
    else:
        raise ValueError(f"Unknown teacher_agg: {teacher_agg}")

    return teacher_avg, teacher_mask, teacher_count


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    lengths_list,
    num_teachers: torch.Tensor,
    temperature: float = 2.0,
    blank_index: int = 0,
    blank_prob_threshold: float = 0.95,
    nonblank_mass_weight: float = 0.0,
    pl_ce_weight: float = 0.0,
    pl_conf_threshold: float = 0.0,
    pl_nonblank_mass_min: float = 0.0,
    pl_label_smoothing: float = 0.0,
    teacher_agg: str = "prob_mean",
    eps: float = 1e-8,
):
    """
    Blank-aware CTC knowledge distillation loss.

    - Averages teacher probabilities across available teachers per frame
    - Ignores frames dominated by blank (teacher blank prob >= threshold)
    - Renormalizes distributions over non-blank tokens before KL

    Args:
        student_logits: [B, Ls, V]
        teacher_logits: [B, T, Lt, V] (padded along T and Lt)
        lengths_list: list of length B; each element is a tensor [Ti] of per-teacher valid lengths
        num_teachers: [B] tensor with number of teachers per sample
        temperature: distillation temperature
        blank_index: index of CTC blank token in vocab
        blank_prob_threshold: frames with teacher blank prob >= thr are ignored
        eps: numerical stability constant

    Returns:
        Tuple (loss, valid_frame_ratio)
    """
    device = student_logits.device
    B, Ls, V = student_logits.shape
    _, Tch, Lt, Vt = teacher_logits.shape
    assert V == Vt, "Student and teacher vocab size must match"

    # Temperature-scaled probabilities/log-probabilities for student
    student_probs = F.softmax(student_logits / temperature, dim=-1)  # [B, Ls, V]

    # Aggregate teacher distributions and masks
    teacher_avg, teacher_mask, teacher_count = aggregate_teacher_distributions(
        teacher_logits,
        lengths_list,
        num_teachers,
        temperature=temperature,
        teacher_agg=teacher_agg,
        eps=eps,
    )

    # Align sequence lengths
    min_len = min(Ls, Lt)
    student_probs = student_probs[:, :min_len, :]
    teacher_avg = teacher_avg[:, :min_len, :]
    teacher_count = teacher_count[:, :min_len]

    # Frame mask candidates and renormalization depend on whether a CTC blank exists
    use_ctc_blank = (blank_index is not None) and (blank_index >= 0) and (blank_index < V)
    if use_ctc_blank:
        teacher_blank = teacher_avg[..., blank_index]  # [B, min_len]
        # Option 1: averaged teacher blank prob below threshold
        valid_by_blank = teacher_blank < blank_prob_threshold  # [B, min_len]
        # Option 2: any teacher predicts a non-blank token at this frame
        # Use raw logits to get per-teacher argmax token ids
        teacher_pred = teacher_logits.argmax(dim=-1)  # [B, T, Lt]
        any_nonblank = ((teacher_pred != blank_index) & teacher_mask)  # [B, T, Lt]
        any_nonblank = any_nonblank.any(dim=1)[:, :min_len]  # [B, min_len]
        # Final valid mask: has at least one teacher frame and either condition
        valid_frames = (teacher_count[:, :min_len] > 0) & (valid_by_blank | any_nonblank)  # [B, min_len]

        # Renormalize over non-blank tokens
        nonblank_mask = torch.ones(V, dtype=torch.bool, device=device)
        nonblank_mask[blank_index] = False

        student_nb = student_probs[..., nonblank_mask]  # [B, min_len, V-1]
        teacher_nb = teacher_avg[..., nonblank_mask]    # [B, min_len, V-1]

        student_nb = student_nb / student_nb.sum(dim=-1, keepdim=True).clamp_min(eps)
        teacher_nb = teacher_nb / teacher_nb.sum(dim=-1, keepdim=True).clamp_min(eps)
    else:
        # No CTC blank (decoder logits). Use full-vocab KL and consider frames valid when any teacher present.
        teacher_blank = torch.zeros((B, min_len), device=device)
        valid_frames = (teacher_count[:, :min_len] > 0)
        student_nb = student_probs  # [B, min_len, V]
        # Avoid degenerate BOS/EOS training at the first token: suppress specials at t=0
        if teacher_avg.size(1) > 0:
            specials = (0, 1, 2)  # <unk>, <s>, </s>
            for sp in specials:
                if sp < V:
                    teacher_avg[:, 0, sp] = 0.0
            # Renormalize
            teacher_avg[:, 0, :] = teacher_avg[:, 0, :] / teacher_avg[:, 0, :].sum(dim=-1, keepdim=True).clamp_min(eps)
        teacher_nb = teacher_avg    # [B, min_len, V]

    # KL divergence per frame
    kl_per_frame = F.kl_div((student_nb + eps).log(), teacher_nb, reduction='none').sum(dim=-1)  # [B, min_len]

    # Mask and average across valid frames
    valid_frames_f = valid_frames.to(student_nb.dtype)
    total_valid = valid_frames_f.sum().clamp_min(1.0)
    loss = (kl_per_frame * valid_frames_f).sum() / total_valid

    # Optional auxiliary 1: match blank mass between teacher and student (only with CTC blank)
    if nonblank_mass_weight > 0.0 and use_ctc_blank:
        student_blank = student_probs[..., blank_index]  # [B, min_len]
        mse_per_frame = (student_blank - teacher_blank) ** 2  # [B, min_len]
        mse_loss = (mse_per_frame * valid_frames_f).sum() / total_valid
        loss = loss + nonblank_mass_weight * mse_loss

        # Prevent student from saturating at blank: also match non-blank mass implicitly
        # (since 1 - blank_mass)

    # Optional auxiliary 2: pseudo-label CE on teacher argmax OVER FULL VOCAB (non-blank target)
    if pl_ce_weight > 0.0:
        with torch.no_grad():
            if use_ctc_blank:
                nb_ids = torch.arange(V, device=device)[nonblank_mask]
                teacher_nb_prob, tgt_nb_idx = teacher_nb.max(dim=-1)
                tgt_full_ids = nb_ids[tgt_nb_idx]
                teacher_nonblank_mass = 1.0 - teacher_blank
            else:
                # Full-vocab max
                teacher_nb_prob, tgt_full_ids = teacher_nb.max(dim=-1)
                teacher_nonblank_mass = torch.ones_like(teacher_nb_prob)
            ce_mask = valid_frames.clone()
            if pl_nonblank_mass_min > 0.0:
                ce_mask = ce_mask & (teacher_nonblank_mass >= pl_nonblank_mass_min)
            if pl_conf_threshold > 0.0:
                ce_mask = ce_mask & (teacher_nb_prob >= pl_conf_threshold)
        student_logits_slice = student_logits[:, :min_len, :]
        ce_flat = F.cross_entropy(
            student_logits_slice.reshape(-1, V),
            tgt_full_ids.reshape(-1),
            reduction='none',
            label_smoothing=float(pl_label_smoothing) if pl_label_smoothing > 0.0 else 0.0,
        ).reshape(B, min_len)
        ce_mask_f = ce_mask.to(ce_flat.dtype)
        ce_denom = ce_mask_f.sum().clamp_min(1.0)
        ce_loss = (ce_flat * ce_mask_f).sum() / ce_denom
        loss = loss + pl_ce_weight * ce_loss

    # Metric: valid-frame ratio over considered timesteps
    valid_frame_ratio = (valid_frames_f.mean()).item()

    # Scale by T^2 per KD convention
    return loss * (temperature ** 2), valid_frame_ratio


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    temperature=2.0,
    grad_clip=1.0,
    blank_index=0,
    blank_prob_threshold=0.95,
    nonblank_mass_weight=0.0,
    pl_ce_weight=0.0,
    pl_conf_threshold: float = 0.0,
    pl_nonblank_mass_min: float = 0.0,
    pl_label_smoothing: float = 0.0,
    teacher_agg: str = "prob_mean",
):
    """
    Train for one epoch.
    
    Args:
        model: Student model
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (called per batch)
        device: Device to train on
        temperature: Distillation temperature
        grad_clip: Gradient clipping max norm
    
    Returns:
        avg_loss: Average loss over epoch
    """
    model.train()
    total_loss = 0.0
    total_vfr = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        try:
            # Get data
            audio = batch['audio'].to(device)  # [batch, max_audio_len]
            audio_lengths = batch['audio_lengths'].to(device)
            teacher_logits = batch['teacher_logits'].to(device)  # [batch, num_teachers, seq_len, vocab]
            
            # Normalize audio lengths to [0, 1]
            audio_lens_norm = audio_lengths.float() / audio_lengths.max().float()
            
            # Forward pass through student (encoder + CTC head)
            encoder_out, ctc_logits = model(audio, audio_lens_norm)

            # Choose student logits source based on KD type
            if blank_index < 0:
                # Decoder KD: build teacher-forced targets from aggregated teacher distributions
                with torch.no_grad():
                    teacher_avg, _, _ = aggregate_teacher_distributions(
                        teacher_logits,
                        lengths_list=batch['lengths'],
                        num_teachers=batch['num_teachers'],
                        temperature=temperature,
                        teacher_agg=teacher_agg,
                    )  # [B, Lt, V]
                    # Build teacher-forced inputs by selecting argmax per step,
                    # but suppress special tokens at the first position to avoid
                    # degenerate EOS/BOS starts.
                    teacher_for_input = teacher_avg.clone()
                    if teacher_for_input.size(1) > 0:
                        # Suppress blank(0), BOS(1), EOS(2) at position 0
                        for sp in (0, 1, 2):
                            if sp < teacher_for_input.size(-1):
                                teacher_for_input[:, 0, sp] = 0.0
                    tgt_ids = teacher_for_input.argmax(dim=-1)  # [B, Lt]
                    # Shift right with BOS=1
                    bos = torch.ones(tgt_ids.size(0), 1, dtype=torch.long, device=tgt_ids.device)
                    decoder_inputs = torch.cat([bos, tgt_ids[:, :-1]], dim=1)  # [B, Lt]
                # Run decoder with teacher-forced inputs
                decoder_logits = model.forward_decoder(encoder_out, decoder_inputs, wav_lens=audio_lens_norm)
                student_logits = decoder_logits  # [B, Lt, V]
            else:
                # CTC KD: use CTC head
                student_logits = ctc_logits  # [B, Ls, V]
            
            # Compute distillation loss (CTC-aware if blank_index>=0, else decoder KD over full vocab)
            loss, vfr = distillation_loss(
                student_logits,
                teacher_logits,
                lengths_list=batch['lengths'],
                num_teachers=batch['num_teachers'],
                temperature=temperature,
                blank_index=blank_index,
                blank_prob_threshold=blank_prob_threshold,
                nonblank_mass_weight=nonblank_mass_weight,
                pl_ce_weight=pl_ce_weight,
                pl_conf_threshold=pl_conf_threshold,
                pl_nonblank_mass_min=pl_nonblank_mass_min,
                pl_label_smoothing=pl_label_smoothing,
                teacher_agg=teacher_agg,
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            # Update learning rate BEFORE optimizer step to avoid large first-batch LR
            scheduler(optimizer)
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            total_vfr += vfr
            num_batches += 1
            
            # Update progress bar with current LR
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'vfr': f'{(vfr*100):.1f}%', 'lr': f'{current_lr:.6f}'})
            
        except Exception as e:
            print(f"\n⚠️  Error in batch: {e}")
            continue
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    avg_vfr = total_vfr / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_vfr


def validate(
    model,
    val_loader,
    device,
    temperature=2.0,
    blank_index=0,
    blank_prob_threshold=0.95,
    nonblank_mass_weight=0.0,
    pl_ce_weight=0.0,
    pl_conf_threshold: float = 0.0,
    pl_nonblank_mass_min: float = 0.0,
    pl_label_smoothing: float = 0.0,
    teacher_agg: str = "prob_mean",
):
    """
    Validate model.
    
    Args:
        model: Student model
        val_loader: Validation data loader
        device: Device
        temperature: Distillation temperature
    
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    total_vfr = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            try:
                # Get data
                audio = batch['audio'].to(device)
                audio_lengths = batch['audio_lengths'].to(device)
                teacher_logits = batch['teacher_logits'].to(device)
                
                # Normalize audio lengths
                audio_lens_norm = audio_lengths.float() / audio_lengths.max().float()
                
                # Forward pass
                encoder_out, ctc_logits = model(audio, audio_lens_norm)
                if blank_index < 0:
                    # Decoder KD validation path
                    with torch.no_grad():
                        teacher_avg, _, _ = aggregate_teacher_distributions(
                            teacher_logits,
                            lengths_list=batch['lengths'],
                            num_teachers=batch['num_teachers'],
                            temperature=temperature,
                            teacher_agg=teacher_agg,
                        )
                        teacher_for_input = teacher_avg.clone()
                        if teacher_for_input.size(1) > 0:
                            for sp in (0, 1, 2):
                                if sp < teacher_for_input.size(-1):
                                    teacher_for_input[:, 0, sp] = 0.0
                        tgt_ids = teacher_for_input.argmax(dim=-1)
                        bos = torch.ones(tgt_ids.size(0), 1, dtype=torch.long, device=tgt_ids.device)
                        decoder_inputs = torch.cat([bos, tgt_ids[:, :-1]], dim=1)
                    decoder_logits = model.forward_decoder(encoder_out, decoder_inputs, wav_lens=audio_lens_norm)
                    student_logits = decoder_logits
                else:
                    student_logits = ctc_logits
                
                # Compute loss (blank-aware CTC KD)
                loss, vfr = distillation_loss(
                    student_logits,
                    teacher_logits,
                    lengths_list=batch['lengths'],
                    num_teachers=batch['num_teachers'],
                    temperature=temperature,
                    blank_index=blank_index,
                    blank_prob_threshold=blank_prob_threshold,
                    nonblank_mass_weight=nonblank_mass_weight,
                    pl_ce_weight=pl_ce_weight,
                    pl_conf_threshold=pl_conf_threshold,
                    pl_nonblank_mass_min=pl_nonblank_mass_min,
                    pl_label_smoothing=pl_label_smoothing,
                    teacher_agg=teacher_agg,
                )
                
                total_loss += loss.item()
                total_vfr += vfr
                num_batches += 1
                
            except Exception as e:
                print(f"\n⚠️  Error in validation batch: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    avg_vfr = total_vfr / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_vfr


def main():
    parser = argparse.ArgumentParser(description='Train Student Conformer via Knowledge Distillation')
    
    # Data arguments
    parser.add_argument('--held_out', type=str, default='M08', help='Held-out speaker for testing')
    parser.add_argument('--matching_mode', type=str, default='partial', 
                        choices=['strict', 'partial', 'all'], help='Utterance matching mode')
    parser.add_argument('--min_teachers', type=int, default=10, 
                        help='Minimum teachers for partial matching')
    parser.add_argument('--exclude_speakers', type=str, nargs='+', default=None,
                        help='Speakers to exclude (e.g., M01)')
    parser.add_argument('--include_held_out_in_training', action='store_true',
                        help='DEBUG: Include held-out speaker in training (for debugging domain shift vs training issues)')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=144, help='Model dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=8, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ffn', type=int, default=1024, help='FFN dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--student_backbone', type=str, default='sb', choices=['sb', 'nemo'], help='Student backbone: sb (SpeechBrain-based) or nemo (NeMo pretrained encoder)')
    parser.add_argument('--nemo_model_name', type=str, default='nvidia/stt_en_conformer_ctc_small', help='NeMo pretrained model name (used when student_backbone=nemo)')
    parser.add_argument('--freeze_nemo_preprocessor', action='store_true', help='Freeze NeMo preprocessor (when using NeMo backbone)')
    parser.add_argument('--freeze_nemo_encoder', action='store_true', help='Freeze NeMo encoder (when using NeMo backbone)')
    
    # Training arguments
    parser.add_argument('--temperature', type=float, default=2.0, help='Distillation temperature')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Warmup steps for scheduler')
    parser.add_argument('--teacher_logits_type', type=str, default='decoder', choices=['ctc', 'decoder'],
                        help='Which teacher logits to distill from (NOTE: use CTC for NeMo backbone due to feature space mismatch)')
    parser.add_argument('--ctc_blank_index', type=int, default=0, help='Index of CTC blank token')
    parser.add_argument('--blank_prob_threshold', type=float, default=0.95, help='Ignore frames with teacher blank prob >= threshold')
    parser.add_argument('--nonblank_mass_weight', type=float, default=0.0, help='Weight for matching blank mass (MSE) between teacher and student')
    parser.add_argument('--pl_ce_weight', type=float, default=0.0, help='Weight for pseudo-label CE on non-blank targets')
    parser.add_argument('--pl_conf_threshold', type=float, default=0.0, help='Minimum teacher top non-blank prob to include a frame in PL CE')
    parser.add_argument('--pl_nonblank_mass_min', type=float, default=0.0, help='Minimum teacher non-blank mass to include a frame in PL CE')
    parser.add_argument('--pl_label_smoothing', type=float, default=0.0, help='Label smoothing for PL CE (0..0.2 recommended)')
    parser.add_argument('--teacher_agg', type=str, default='prob_mean', choices=['prob_mean','logprob_mean','logit_mean'], help='Aggregation of teacher distributions across teachers')
    
    # Curriculum learning arguments
    parser.add_argument('--curriculum_schedule', type=str, default='none', 
                        choices=['none', 'linear', 'sqrt', 'step'],
                        help='Curriculum learning schedule (none=no curriculum, linear/sqrt/step=progressive)')
    parser.add_argument('--curriculum_scores', type=str, default=None,
                        help='Path to JSON file with curriculum difficulty scores')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()

    # Sensible default: exclude outlier speaker M01 unless explicitly overridden
    if args.exclude_speakers is None:
        args.exclude_speakers = []
    
    # DEBUG mode: Include held-out speaker in training
    if args.include_held_out_in_training:
        print("\n🔧 DEBUG MODE: Including held-out speaker in training")
        print(f"   Speaker '{args.held_out}' will be in BOTH training and test sets")
        print("   This tests if model can fit training data (domain shift vs training issue)")
        print("   Expected: High accuracy on test if training works\n")
        # Don't auto-switch to CTC in debug mode - let user control
    else:
        # CRITICAL: Force CTC mode when using NeMo backbone (only in normal mode)
        # Reason: Teacher decoder logits were computed from SpeechBrain encoders, which produce
        # different feature representations than NeMo. Decoder-KD with mismatched feature spaces
        # leads to complete training failure (0% WRA). CTC distillation works because it only
        # requires frame-level alignment, not feature space compatibility.
        #if args.student_backbone == 'nemo' and args.teacher_logits_type == 'decoder':
          # print("\n⚠️  WARNING: Decoder-based KD is incompatible with NeMo backbone!")
           # print("   Reason: Feature space mismatch between NeMo encoder and teacher decoders")
           # print("   Automatically switching to --teacher_logits_type ctc")
           # print("   (Teacher decoder logits were computed from SpeechBrain encoders)\n")
            args.teacher_logits_type = 'decoder'
    
    # Create save directory
    save_dir = Path(args.save_dir) / f"student_{args.held_out}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n" + "="*80)
    print(f"🎓 Training Student Conformer - {args.held_out} Fold")
    print("="*80)
    print(f"🔧 KD source: {args.teacher_logits_type} logits")
    
    # Create dataset
    print(f"\n📊 Loading training dataset...")
    
    # Check if curriculum learning is enabled
    use_curriculum = args.curriculum_schedule != 'none' and args.curriculum_scores is not None
    if use_curriculum:
        print(f"📚 Curriculum learning enabled: {args.curriculum_schedule} schedule")
        print(f"📚 Using difficulty scores from: {args.curriculum_scores}")
        if not os.path.exists(args.curriculum_scores):
            raise FileNotFoundError(f"Curriculum scores file not found: {args.curriculum_scores}")
    else:
        if args.curriculum_schedule != 'none':
            print("⚠️  Curriculum schedule specified but no scores file provided - using baseline (no curriculum)")
        print("📚 Baseline training: using all data from start")
    
    # In debug mode, we want M08 logits from other speakers (M08 becomes a teacher, not held-out)
    if args.include_held_out_in_training:
        # Don't hold out M08 - use it as one of the teachers
        # Pick a different speaker to "hold out" that we won't use
        # This way M08's own logits will be in the training set
        print(f"   DEBUG: Using '{args.held_out}' as a TEACHER (not held-out)")
        train_dataset = LogitEnsembleDataset(
            held_out_speaker=None,  # No speaker held out, use all
            split="train",
            use_decoder_logits=(args.teacher_logits_type == 'decoder'),
            matching_mode=args.matching_mode,
            min_teachers=args.min_teachers,
            exclude_speakers=args.exclude_speakers,
            curriculum_scores_file=args.curriculum_scores if use_curriculum else None
        )
    else:
        train_dataset = LogitEnsembleDataset(
            held_out_speaker=args.held_out,
            split="train",
            use_decoder_logits=(args.teacher_logits_type == 'decoder'),
            matching_mode=args.matching_mode,
            min_teachers=args.min_teachers,
            exclude_speakers=args.exclude_speakers,
            curriculum_scores_file=args.curriculum_scores if use_curriculum else None
        )
    
    # Create validation dataset (use a subset for quick validation)
    # For now, we'll use a small portion of training data
    # TODO: Create proper validation split
    val_size = min(100, len(train_dataset) // 10)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Store the full train dataset for curriculum learning
    # (we need access to the original dataset, not the random_split wrapper)
    full_train_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
    
    # Create data loaders
    # Note: For curriculum learning, we'll recreate the train_loader each epoch
    # For now, create the initial loader (will be replaced in training loop if using curriculum)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_logits,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_logits,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Estimate steps per epoch for scheduler diagnostics
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    # Adjust warmup if it's too large (e.g., larger than total steps)
    original_warmup = args.warmup_steps
    max_reasonable_warmup = max(100, int(0.1 * max(1, total_steps)))
    if args.warmup_steps > max_reasonable_warmup:
        print("\n⚠️  Warmup steps too large for dataset; adjusting.")
        print(f"   Requested warmup: {args.warmup_steps}, Total steps: {total_steps}, 10% total: {max_reasonable_warmup}")
        args.warmup_steps = max_reasonable_warmup
        print(f"   Using warmup_steps={args.warmup_steps} instead.")

    print(f"\n🧭 Steps per epoch: {steps_per_epoch}")
    print(f"🧭 Total steps (approx): {total_steps}")
    print(f"🧭 Warmup steps: {args.warmup_steps} ({(args.warmup_steps/ max(1,total_steps))*100:.1f}% of total)")

    # Create model
    print(f"\n🏗️  Creating student model ({'NeMo' if args.student_backbone=='nemo' else 'SpeechBrain'})...")
    if args.student_backbone == 'nemo':
        if not _HAS_NEMO:
            raise RuntimeError("Requested student_backbone='nemo' but NeMo is not available. Install with: pip install \"nemo_toolkit[all]\"")
        model = NeMoHybridStudent(
            nemo_model_name=args.nemo_model_name,
            vocab_size=5000,
            num_decoder_layers=args.num_decoder_layers,
            nhead=args.nhead,
            d_ffn=args.d_ffn,
            dropout=args.dropout,
            freeze_nemo_preprocessor=args.freeze_nemo_preprocessor,
            freeze_nemo_encoder=args.freeze_nemo_encoder,
            device=torch.device(args.device) if isinstance(args.device, str) else args.device,
        )
    else:
        model = StudentConformer(
            vocab_size=5000,
            d_model=args.d_model,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            nhead=args.nhead,
            d_ffn=args.d_ffn,
            dropout=args.dropout
        ).to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Optimizer (AdamW with warmup)
    # Note: NoamScheduler uses lr = lr_initial * (model_size^-0.5) * min(step^-0.5, step * warmup^-1.5)
    # To target a desired PEAK LR (= args.lr) at step==warmup_steps, set:
    #   lr_initial = peak_lr * sqrt(model_size * warmup_steps)
    model_size = getattr(model, 'd_model', args.d_model)
    peak_lr = args.lr
    lr_initial = peak_lr * math.sqrt(model_size * args.warmup_steps)
    print(f"\n🎯 Desired peak LR: {peak_lr:.3e}")
    print(f"🧮 Computed base lr_initial for Noam: {lr_initial:.3e}")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr_initial, 
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler (Noam scheduler with model_size parameter)
    from speechbrain.nnet.schedulers import NoamScheduler
    scheduler = NoamScheduler(
        lr_initial=lr_initial,
        n_warmup_steps=args.warmup_steps,
        model_size=model_size  # Use actual model size of selected backbone
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\n📂 Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✅ Resumed from epoch {start_epoch}")
        # Guard against no-op training when --epochs <= start_epoch
        if args.epochs <= start_epoch:
            print("\n⚠️  Provided --epochs is less than or equal to the resumed epoch. Adjusting epochs to continue training.")
            args.epochs = start_epoch + 1
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"   Temperature: {args.temperature}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {args.batch_size}")
    blank_index_for_loss = args.ctc_blank_index if args.teacher_logits_type == 'ctc' else -1
    if args.teacher_logits_type == 'ctc':
        print(f"   CTC blank index: {args.ctc_blank_index}")
    else:
        print(f"   Distillation source: decoder logits (no CTC blank masking)")
    print(f"   Blank prob threshold: {args.blank_prob_threshold}")
    print(f"   Nonblank mass weight: {args.nonblank_mass_weight}")
    print(f"   PL CE weight: {args.pl_ce_weight}")
    print(f"   PL conf threshold: {args.pl_conf_threshold}")
    print(f"   PL nonblank mass min: {args.pl_nonblank_mass_min}")
    print(f"   PL label smoothing: {args.pl_label_smoothing}")
    print(f"   Teacher aggregation: {args.teacher_agg}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")

        # Curriculum learning: adjust data subset based on competence
        if use_curriculum:
            competence = get_curriculum_competence(epoch, args.epochs, args.curriculum_schedule)
            curriculum_indices = full_train_dataset.get_curriculum_subset_indices(competence)
            
            # Map curriculum indices to the train_dataset indices (accounting for train/val split)
            # train_dataset contains a subset of full_train_dataset after random_split
            # We need to filter curriculum_indices to only include those in train_dataset
            if hasattr(train_dataset, 'indices'):
                # train_dataset is a Subset from random_split
                train_subset_indices = set(train_dataset.indices)
                # Filter curriculum indices to those in our training subset
                valid_curriculum_indices = [idx for idx in curriculum_indices if idx in train_subset_indices]
                # Map to positions in train_dataset (0 to len(train_dataset)-1)
                idx_to_pos = {orig_idx: pos for pos, orig_idx in enumerate(train_dataset.indices)}
                curriculum_positions = [idx_to_pos[idx] for idx in valid_curriculum_indices if idx in idx_to_pos]
            else:
                # train_dataset is the full dataset
                curriculum_positions = curriculum_indices
            
            # Create sampler for curriculum subset
            train_sampler = SubsetRandomSampler(curriculum_positions)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                collate_fn=collate_logits,
                num_workers=args.num_workers,
                pin_memory=True if args.device == 'cuda' else False
            )
            
            print(f"📚 Curriculum competence: {competence:.3f} ({len(curriculum_positions)}/{len(train_dataset)} samples)")
        else:
            # Baseline: recreate loader with shuffle for randomness each epoch
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_logits,
                num_workers=args.num_workers,
                pin_memory=True if args.device == 'cuda' else False
            )

        # Get current learning rate from optimizer
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate (start of epoch): {current_lr:.6e}")

        # Train (scheduler is called inside train_epoch per batch)
        train_loss, train_vfr = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            args.device,
            args.temperature,
            args.grad_clip,
            blank_index_for_loss,
            args.blank_prob_threshold,
            args.nonblank_mass_weight,
            args.pl_ce_weight,
            args.pl_conf_threshold,
            args.pl_nonblank_mass_min,
            args.pl_label_smoothing,
            args.teacher_agg,
        )
        print(f"📊 Train Loss: {train_loss:.4f} | Valid-frame ratio: {train_vfr*100:.1f}%")

        # Get LR after training
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate (end of epoch): {current_lr:.6e}")

        # Validate
        val_loss, val_vfr = validate(
            model,
            val_loader,
            args.device,
            args.temperature,
            blank_index_for_loss,
            args.blank_prob_threshold,
            args.nonblank_mass_weight,
            args.pl_ce_weight,
            args.pl_conf_threshold,
            args.pl_nonblank_mass_min,
            args.pl_label_smoothing,
            args.teacher_agg,
        )
        print(f"📊 Val Loss: {val_loss:.4f} | Valid-frame ratio: {val_vfr*100:.1f}%")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': vars(args)
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, save_dir / 'latest.pt')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, save_dir / 'best.pt')
            torch.save(model.state_dict(), save_dir / f'student_{args.held_out}_best.pt')
            print(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    print(f"\n{'='*80}")
    print("🎉 Training Complete!")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
