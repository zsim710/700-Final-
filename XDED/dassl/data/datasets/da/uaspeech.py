import os
import os.path as osp
import json
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class UASpeechDataset(DatasetBase):
    """UASpeech Dataset for Audio Domain Adaptation.
    
    Statistics:
        - Dysarthric speech recognition dataset
        - 14 speakers with varying severity levels
        - Isolated word recognition task
        - Domains: Each speaker is treated as a separate domain
        
    Speakers:
        - F03, F04, F05, M01, M04, M05, M07, M08, M09, M10, M11, M12, M14, M16
    """
    
    dataset_dir = 'uaspeech'
    # All available speakers as domains
    domains = ['F03', 'F04', 'F05', 'M01', 'M04', 'M05', 'M07', 'M08', 
               'M09', 'M10', 'M11', 'M12', 'M14', 'M16']
    
    def __init__(self, cfg):
        # Use the existing audio data paths
        self.partition_dir = '/home/zsim710/partitions/uaspeech/by_speakers'
        self.audio_root = '/home/zsim710/XDED/speechbrain/datasets/UASpeech'
        
        # Check that the required domains are valid
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        
        # Create word vocabulary from ALL domains (source + target) to ensure coverage
        all_speakers = list(set(cfg.DATASET.SOURCE_DOMAINS + (cfg.DATASET.TARGET_DOMAINS or [])))
        self.word_to_label = self._build_vocabulary(all_speakers)
        self.label_to_word = {v: k for k, v in self.word_to_label.items()}
        
        print(f"📚 Built vocabulary with {len(self.word_to_label)} words")
        print(f"🎯 Sample words: {list(self.word_to_label.keys())[:10]}")
        
        # Load training data from source domains
        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split='train')
        
        # Load target data (can be same as source for domain adaptation)
        if cfg.DATASET.TARGET_DOMAINS:
            train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, split='train') 
            test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split='test')
        else:
            # No target domains specified, use source domains for testing
            train_u = []
            test = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split='test')
        
        super().__init__(train_x=train_x, train_u=train_u, test=test)
        
        print(f"🎵 Dataset created:")
        print(f"   - Training samples: {len(train_x)}")
        print(f"   - Unlabeled samples: {len(train_u)}")  
        print(f"   - Test samples: {len(test)}")
    
    def _build_vocabulary(self, speaker_list):
        """Build word vocabulary from CSV files."""
        all_words = set()
        
        for speaker in speaker_list:
            csv_file = osp.join(self.partition_dir, f"{speaker}.csv")
            if not osp.exists(csv_file):
                print(f"⚠️  Warning: CSV file not found for {speaker}: {csv_file}")
                continue
            
            try:
                df = pd.read_csv(csv_file)
                if 'wrd' in df.columns:
                    # Get words from 'wrd' column and ensure consistent format
                    words = df['wrd'].str.strip().str.upper().unique()
                    all_words.update(words)
                elif 'transcription' in df.columns:
                    # Fallback to 'transcription' column if exists
                    words = df['transcription'].str.strip().str.upper().unique()
                    all_words.update(words)
                else:
                    print(f"⚠️  Warning: No 'wrd' or 'transcription' column in {csv_file}")
            except Exception as e:
                print(f"❌ Error reading {csv_file}: {e}")
        
        # Create word to label mapping
        word_to_label = {word: idx for idx, word in enumerate(sorted(all_words))}
        return word_to_label
    
    def _read_data(self, input_domains, split='train'):
        """Read data for specified domains and split."""
        items = []
        
        for domain_idx, speaker in enumerate(input_domains):
            csv_file = osp.join(self.partition_dir, f"{speaker}.csv")
            
            if not osp.exists(csv_file):
                print(f"⚠️  Warning: CSV file not found for {speaker}: {csv_file}")
                continue
            
            try:
                # Read CSV with correct column names from uaspeech_prepare.py
                # Columns: ["ID", "duration", "wav", "spk_id", "wrd"]
                df = pd.read_csv(csv_file)
                
                # Use all data for now (no train/test split in individual speaker files)
                # Each speaker CSV contains all their data
                for _, row in df.iterrows():
                    try:
                        # Get audio file path (already full path in 'wav' column)
                        if 'wav' in row:
                            audio_path = row['wav']
                        else:
                            print(f"⚠️  Warning: No 'wav' column found in {csv_file}")
                            continue
                        
                        # Get word transcription from 'wrd' column
                        if 'wrd' in row:
                            word = row['wrd'].strip().upper()  # Ensure consistent format
                        else:
                            print(f"⚠️  Warning: No 'wrd' column found in {csv_file}")
                            continue
                        
                        # Get utterance ID
                        if 'ID' in row:
                            utterance_id = row['ID']
                        else:
                            utterance_id = f"{speaker}_{len(items)}"
                        
                        # Convert word to label
                        if word in self.word_to_label:
                            label = self.word_to_label[word]
                            
                            # Create data item
                            item = Datum(
                                impath=audio_path,      # Full path to audio file
                                label=label,            # Word class ID
                                domain=domain_idx,      # Speaker domain ID  
                                classname=word          # Original word text
                            )
                            # Add utterance ID for later reference
                            item.utterance_id = utterance_id
                            items.append(item)
                        else:
                            print(f"⚠️  Warning: Unknown word '{word}' not in vocabulary")
                    
                    except Exception as e:
                        print(f"⚠️  Warning: Error processing row in {csv_file}: {e}")
                        continue
                        
            except Exception as e:
                print(f"❌ Error processing {speaker} CSV: {e}")
        
        print(f"📊 Loaded {len(items)} samples for {split} from domains: {input_domains}")
        return items