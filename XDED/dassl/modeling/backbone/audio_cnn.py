"""
Audio backbone for speech recognition using mel-spectrograms and CNN.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


@BACKBONE_REGISTRY.register()
class AudioCNN(Backbone):
    """Simple CNN backbone for audio processing.
    
    Pipeline:
    1. Raw audio [B, T] → Mel-spectrogram [B, 1, n_mels, T']  
    2. CNN layers → Features [B, feature_dim]
    
    This treats mel-spectrograms as grayscale images for CNN processing.
    """
    
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=512, hop_length=160, 
                 win_length=400, feature_dim=256, **kwargs):
        super().__init__()
        
        # Audio preprocessing parameters
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.feature_dim = feature_dim
        
        # Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=True,
            pad_mode='reflect',
            power=2.0,
            norm='slaney'
        )
        
        # Convert to log scale (dB)
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
        
        # CNN layers for feature extraction
        self.conv_layers = nn.Sequential(
            # Input: [B, 1, 80, T'] - like grayscale images
            
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [B, 32, 40, T'/2]
            
            # Block 2  
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [B, 64, 20, T'/4]
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [B, 128, 10, T'/8]
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 256, 1, 1] - global average pooling
        )
        
        # Final feature layer
        self.feature_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self._out_features = feature_dim
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw audio waveforms [B, T] or [B, 1, T]
        
        Returns:
            torch.Tensor: Audio features [B, feature_dim]
        """
        # Handle different input shapes
        if len(x.shape) == 3:  # [B, 1, T]
            x = x.squeeze(1)  # [B, T]
        elif len(x.shape) == 1:  # [T] - single sample
            x = x.unsqueeze(0)  # [1, T]
        
        # 1. Convert to mel-spectrogram
        mel_spec = self.mel_transform(x)  # [B, n_mels, T']
        
        # 2. Convert to log scale
        log_mel = self.amplitude_to_db(mel_spec)  # [B, n_mels, T']
        
        # 3. Add channel dimension (like grayscale images)
        log_mel = log_mel.unsqueeze(1)  # [B, 1, n_mels, T']
        
        # 4. CNN feature extraction
        conv_features = self.conv_layers(log_mel)  # [B, 256, 1, 1]
        
        # 5. Flatten for linear layer
        conv_features = conv_features.view(conv_features.size(0), -1)  # [B, 256]
        
        # 6. Final feature transformation
        features = self.feature_layer(conv_features)  # [B, feature_dim]
        
        return features
    
    @property 
    def out_features(self):
        return self._out_features


@BACKBONE_REGISTRY.register()
def audio_cnn(**kwargs):
    """Factory function for AudioCNN backbone."""
    return AudioCNN(**kwargs)


@BACKBONE_REGISTRY.register()  
def audio_cnn_small(**kwargs):
    """Smaller version of AudioCNN for faster experimentation."""
    return AudioCNN(feature_dim=128, **kwargs)


@BACKBONE_REGISTRY.register()
def audio_cnn_large(**kwargs):
    """Larger version of AudioCNN for better performance.""" 
    return AudioCNN(feature_dim=512, **kwargs)