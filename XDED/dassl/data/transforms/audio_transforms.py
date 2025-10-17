"""
Audio-specific transforms for speech processing.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import random
import numpy as np


class LoadAudio:
    """Load audio file and convert to tensor."""
    
    def __init__(self, sample_rate=16000, max_length=16000):
        self.sample_rate = sample_rate
        self.max_length = max_length  # 1 second of audio at 16kHz
    
    def __call__(self, audio_path):
        """Load audio from file path."""
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Pad or truncate to fixed length
            if waveform.shape[1] > self.max_length:
                # Truncate: take center portion
                start = (waveform.shape[1] - self.max_length) // 2
                waveform = waveform[:, start:start + self.max_length]
            elif waveform.shape[1] < self.max_length:
                # Pad with zeros
                pad_length = self.max_length - waveform.shape[1]
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
            
            return waveform.squeeze(0)  # Return [T] instead of [1, T]
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return silence if loading fails
            return torch.zeros(self.max_length)


class AudioNormalize:
    """Normalize audio to [-1, 1] range."""
    
    def __call__(self, audio):
        """Normalize audio amplitude."""
        if torch.max(torch.abs(audio)) > 0:
            return audio / torch.max(torch.abs(audio))
        return audio


class AudioNoiseAugment:
    """Add random noise to audio."""
    
    def __init__(self, noise_factor=0.005):
        self.noise_factor = noise_factor
    
    def __call__(self, audio):
        if random.random() > 0.5:  # 50% chance to apply
            noise = torch.randn_like(audio) * self.noise_factor
            return audio + noise
        return audio


class AudioTimeStretch:
    """Apply time stretching/compression."""
    
    def __init__(self, stretch_factor_range=(0.8, 1.2)):
        self.stretch_factor_range = stretch_factor_range
    
    def __call__(self, audio):
        if random.random() > 0.5:  # 50% chance to apply
            stretch_factor = random.uniform(*self.stretch_factor_range)
            
            # Simple implementation: resample to achieve time stretching effect
            original_length = len(audio)
            stretched_length = int(original_length / stretch_factor)
            
            # Interpolate to new length
            indices = torch.linspace(0, original_length - 1, stretched_length)
            stretched = torch.nn.functional.interpolate(
                audio.unsqueeze(0).unsqueeze(0), 
                size=stretched_length, 
                mode='linear',
                align_corners=True
            ).squeeze()
            
            # Pad or truncate back to original length
            if len(stretched) > original_length:
                start = (len(stretched) - original_length) // 2
                return stretched[start:start + original_length]
            elif len(stretched) < original_length:
                pad_length = original_length - len(stretched)
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                return torch.nn.functional.pad(stretched, (pad_left, pad_right))
            
            return stretched
        return audio


class AudioVolumeAugment:
    """Random volume adjustment."""
    
    def __init__(self, volume_range=(0.5, 2.0)):
        self.volume_range = volume_range
    
    def __call__(self, audio):
        if random.random() > 0.5:  # 50% chance to apply
            volume_factor = random.uniform(*self.volume_range)
            return audio * volume_factor
        return audio


class AudioCompose:
    """Compose multiple audio transforms."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, audio):
        for transform in self.transforms:
            audio = transform(audio)
        return audio


# Standard transform combinations
def get_audio_transforms(mode='train', sample_rate=16000, max_length=16000):
    """Get standard audio transform pipeline."""
    
    base_transforms = [
        LoadAudio(sample_rate=sample_rate, max_length=max_length),
        AudioNormalize()
    ]
    
    if mode == 'train':
        # Add augmentations for training
        augment_transforms = [
            AudioNoiseAugment(noise_factor=0.01),
            AudioVolumeAugment(volume_range=(0.7, 1.3)),
            AudioTimeStretch(stretch_factor_range=(0.9, 1.1))
        ]
        return AudioCompose(base_transforms + augment_transforms)
    else:
        # Only basic transforms for validation/test
        return AudioCompose(base_transforms)