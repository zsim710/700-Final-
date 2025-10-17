"""
Logit-based Dataset for Knowledge Distillation
Pure logit-level distillation without feature extraction.

Loads pre-extracted logits from 14 SA models (teacher models) and
creates a dataset for training a student model via knowledge distillation.
"""

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torchaudio


class LogitEnsembleDataset(Dataset):
    """
    Dataset that loads pre-extracted logits from multiple SA models.
    
    For knowledge distillation, we load:
    - Teacher logits: From 14 SA models (all except held-out speaker)
    - Student target: Ensemble of teacher logits or ground truth labels
    
    Leave-One-Out Strategy:
    - HIGH intelligibility: Hold out M08, train on [F02, F03, M04, M07, M12, M14]
    - MID intelligibility: Hold out M05, train on [F04, M10, M11]  
    - LOW intelligibility: Hold out M16, train on [F05, M09]
    - VERY_LOW intelligibility: Hold out M01, train on [] (use other levels)
    """
    
    def __init__(
        self,
        logit_root_dir="/home/zsim710/XDED/speechbrain/exp_results/logit_extraction",
        held_out_speaker="M08",
        split="train",
        intelligibility_level="HIGH",
        use_decoder_logits=True,
        matching_mode="partial",
        min_teachers=10,
        exclude_speakers=None,
        csv_dir="/home/zsim710/partitions/uaspeech/by_speakers",
        curriculum_scores_file=None
    ):
        """
        Args:
            logit_root_dir: Root directory containing extracted logits
            held_out_speaker: Speaker to hold out for testing
            split: 'train' or 'test'
            intelligibility_level: 'HIGH', 'MID', 'LOW', or 'VERY_LOW'
            use_decoder_logits: If True, use decoder logits; else use CTC logits
            matching_mode: 'strict' (all teachers), 'partial' (>=min_teachers), or 'all' (no matching)
            min_teachers: Minimum number of teachers required for partial matching (default: 10)
            exclude_speakers: List of speakers to exclude from teachers (e.g., ['M01'] for outliers)
            csv_dir: Directory containing CSV files with audio paths
            curriculum_scores_file: Path to JSON file with curriculum difficulty scores (optional)
        """
        self.logit_root_dir = logit_root_dir
        self.held_out_speaker = held_out_speaker
        self.split = split
        self.intelligibility_level = intelligibility_level
        self.use_decoder_logits = use_decoder_logits
        self.matching_mode = matching_mode
        self.min_teachers = min_teachers
        self.exclude_speakers = exclude_speakers or []
        self.csv_dir = csv_dir
        self.curriculum_scores_file = curriculum_scores_file
        
        # Curriculum learning attributes
        self.curriculum_scores = None  # {core_id: difficulty_score}
        self.curriculum_order = None  # Sorted list of core_ids by difficulty (easy->hard)
        
        # Define intelligibility level mapping (corrected)
        self.intelligibility_map = {
            "VERY_LOW": ["M04", "F03", "M12", "M01"],
            "LOW": ["M07", "F02", "M16"],
            "MID": ["M05", "M11", "F04"],
            "HIGH": ["M09", "M14", "M10", "M08", "F05"]
        }
        
        # All 15 speakers
        self.all_speakers = ["F02", "F03", "F04", "F05", "M01", "M04", "M05", 
                            "M07", "M08", "M09", "M10", "M11", "M12", "M14", "M16"]
        
        # Get teacher speakers - ALL speakers except held-out and excluded speakers
        self.teacher_speakers = [
            spk for spk in self.all_speakers
            if spk != held_out_speaker and spk not in self.exclude_speakers
        ]
        
        # For testing, we evaluate on the held-out speaker
        self.test_speaker = held_out_speaker
        
        print(f"\n📊 Initializing LogitEnsembleDataset:")
        print(f"   Held-out Speaker (test): {held_out_speaker}")
        print(f"   Excluded Speakers: {self.exclude_speakers if self.exclude_speakers else 'None'}")
        print(f"   Teacher Speakers (train): {len(self.teacher_speakers)} speakers")
        print(f"   Teachers: {self.teacher_speakers}")
        print(f"   Matching Mode: {matching_mode}")
        if matching_mode == "partial":
            print(f"   Min Teachers Required: {min_teachers}")
        print(f"   Split: {split}")
        
        # Load logits and metadata
        self.teacher_logits = {}  # {speaker: [logits_per_utterance]}
        self.metadata = {}  # {speaker: metadata_dict}
        
        # Core-ID mapping for cross-speaker utterance matching
        self.coreid_to_speakers = {}  # {core_id: {speaker: local_idx}}
        self.utterance_ids = []  # List of core_ids (for train) or (speaker, utt_id) tuples (for test)
        
        # Audio path mappings from CSV files
        self.audio_paths = {}  # {speaker: {utt_id: wav_path}}
        
        if split == "train":
            self._load_teacher_logits()
            self._load_csv_mappings()
            if curriculum_scores_file:
                self._load_curriculum_scores()
        else:  # test
            self._load_test_logits()
            self._load_csv_mappings()
    
    def _load_teacher_logits(self):
        """Load logits from all teacher speakers and build core-ID mapping."""
        print(f"\n🔄 Loading teacher logits from {len(self.teacher_speakers)} speakers...")
        
        for speaker in self.teacher_speakers:
            logit_file = self._get_logit_path(speaker)
            metadata_file = self._get_metadata_path(speaker)
            
            if not os.path.exists(logit_file):
                print(f"   ⚠️  Logit file not found for {speaker}: {logit_file}")
                continue
            
            try:
                # Load logits
                logits = torch.load(logit_file, map_location='cpu')
                self.teacher_logits[speaker] = logits
                
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                self.metadata[speaker] = metadata
                
                # Get utterance IDs (handle different metadata formats)
                utt_ids = metadata.get('utterance_ids') or metadata.get('utterances', [])
                
                print(f"   ✅ Loaded {speaker}: {len(logits)} utterances")
                
                # Build core-ID mapping (remove speaker prefix)
                for local_idx, utt_id in enumerate(utt_ids):
                    # Extract core ID by removing speaker prefix (e.g., "M08_B3_D3_M2" -> "B3_D3_M2")
                    if '_' in utt_id:
                        core_id = utt_id.split('_', 1)[1]
                    else:
                        core_id = utt_id
                    
                    if core_id not in self.coreid_to_speakers:
                        self.coreid_to_speakers[core_id] = {}
                    
                    self.coreid_to_speakers[core_id][speaker] = local_idx
                    
            except Exception as e:
                print(f"   ❌ Error loading {speaker}: {e}")
        
        # Filter core-IDs based on matching mode
        print(f"\n🔍 Filtering utterances by matching mode: {self.matching_mode}")
        
        if self.matching_mode == "strict":
            # Only keep core-IDs present in ALL teachers
            valid_coreids = [
                core_id for core_id, speakers in self.coreid_to_speakers.items()
                if len(speakers) == len(self.teacher_speakers)
            ]
            print(f"   Strict mode: Requiring all {len(self.teacher_speakers)} teachers")
        
        elif self.matching_mode == "partial":
            # Keep core-IDs with at least min_teachers
            valid_coreids = [
                core_id for core_id, speakers in self.coreid_to_speakers.items()
                if len(speakers) >= self.min_teachers
            ]
            print(f"   Partial mode: Requiring ≥{self.min_teachers} teachers")
        
        else:  # "all" - no matching, use all utterances
            valid_coreids = list(self.coreid_to_speakers.keys())
            print(f"   All mode: Using all utterances (no matching)")
        
        self.utterance_ids = valid_coreids
        
        # Calculate statistics
        if valid_coreids:
            avg_teachers = sum(len(self.coreid_to_speakers[cid]) for cid in valid_coreids) / len(valid_coreids)
            print(f"\n✅ Total training samples: {len(self.utterance_ids)}")
            print(f"   Average teachers per utterance: {avg_teachers:.1f}")
        else:
            print(f"\n⚠️  No valid utterances found with current matching criteria!")
    
    def _load_test_logits(self):
        """Load logits from held-out test speaker."""
        print(f"\n🔄 Loading test logits from {self.test_speaker}...")
        
        logit_file = self._get_logit_path(self.test_speaker)
        metadata_file = self._get_metadata_path(self.test_speaker)
        
        if not os.path.exists(logit_file):
            raise FileNotFoundError(f"Test logit file not found: {logit_file}")
        
        # Load test logits
        logits = torch.load(logit_file, map_location='cpu')
        self.teacher_logits[self.test_speaker] = logits
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        self.metadata[self.test_speaker] = metadata
        
        # Store utterance IDs
        for utt_id in range(len(logits)):
            self.utterance_ids.append((self.test_speaker, utt_id))
        
        print(f"   ✅ Loaded {self.test_speaker}: {len(logits)} test utterances")
    
    def _get_logit_path(self, speaker):
        """Get path to logit file for a speaker."""
        logit_type = "decoder_logits" if self.use_decoder_logits else "ctc_logits"
        return os.path.join(
            self.logit_root_dir,
            speaker,
            speaker,
            f"{speaker}_{logit_type}.pt"
        )
    
    def _get_metadata_path(self, speaker):
        """Get path to metadata file for a speaker."""
        return os.path.join(
            self.logit_root_dir,
            speaker,
            speaker,
            f"{speaker}_metadata.json"
        )
    
    def _load_csv_mappings(self):
        """Load CSV files to get utterance_id -> audio_path mappings."""
        print(f"\n📂 Loading audio path mappings from CSVs...")
        
        # For training: load CSVs for all teacher speakers
        # For testing: load CSV for held-out speaker
        speakers_to_load = self.teacher_speakers if self.split == "train" else [self.test_speaker]
        
        for speaker in speakers_to_load:
            csv_path = os.path.join(self.csv_dir, f"{speaker}.csv")
            
            if not os.path.exists(csv_path):
                print(f"   ⚠️  CSV not found for {speaker}: {csv_path}")
                continue
            
            try:
                # Read CSV
                df = pd.read_csv(csv_path)
                
                # Create mapping: utterance_id -> wav_path
                self.audio_paths[speaker] = {}
                for _, row in df.iterrows():
                    utt_id = row['ID']  # Full ID like "F02_B3_CW100_M2"
                    wav_path = row['wav']
                    self.audio_paths[speaker][utt_id] = wav_path
                
                print(f"   ✅ Loaded {speaker}: {len(self.audio_paths[speaker])} audio paths")
                
            except Exception as e:
                print(f"   ❌ Error loading CSV for {speaker}: {e}")
    
    def _load_curriculum_scores(self):
        """Load curriculum difficulty scores from JSON file."""
        print(f"\n📚 Loading curriculum difficulty scores...")
        
        if not os.path.exists(self.curriculum_scores_file):
            raise FileNotFoundError(f"Curriculum scores file not found: {self.curriculum_scores_file}")
        
        try:
            with open(self.curriculum_scores_file, 'r') as f:
                data = json.load(f)
            
            self.curriculum_scores = data.get('utterance_scores', {})
            
            # Verify that we have scores for our utterances
            matched_count = 0
            missing_count = 0
            for core_id in self.utterance_ids:
                # Try to find the score with different possible formats
                score = None
                
                # Try direct match first
                if core_id in self.curriculum_scores:
                    score = self.curriculum_scores[core_id]
                    matched_count += 1
                else:
                    # Try to match with speaker prefix
                    for teacher_speaker in self.teacher_speakers:
                        full_id = f"{teacher_speaker}_{core_id}"
                        if full_id in self.curriculum_scores:
                            # Cache this mapping for future use
                            self.curriculum_scores[core_id] = self.curriculum_scores[full_id]
                            score = self.curriculum_scores[full_id]
                            matched_count += 1
                            break
                
                if score is None:
                    missing_count += 1
            
            print(f"   ✅ Loaded curriculum scores for {matched_count}/{len(self.utterance_ids)} utterances")
            if missing_count > 0:
                print(f"   ⚠️  Missing scores for {missing_count} utterances (will use default difficulty)")
            
            # Create curriculum order (easy to hard)
            self._create_curriculum_order()
            
        except Exception as e:
            print(f"   ❌ Error loading curriculum scores: {e}")
            raise
    
    def _create_curriculum_order(self):
        """Create a sorted list of utterance indices ordered by difficulty (easy -> hard)."""
        if self.curriculum_scores is None:
            print("   ⚠️  No curriculum scores loaded, cannot create curriculum order")
            return
        
        # Create list of (idx, core_id, difficulty_score) tuples
        utterance_difficulties = []
        for idx, core_id in enumerate(self.utterance_ids):
            # Get difficulty score (default to 0.5 if missing)
            score = self.curriculum_scores.get(core_id, 0.5)
            utterance_difficulties.append((idx, core_id, score))
        
        # Sort by difficulty score (ascending = easy to hard)
        utterance_difficulties.sort(key=lambda x: x[2])
        
        # Store sorted indices
        self.curriculum_order = [idx for idx, _, _ in utterance_difficulties]
        
        # Print statistics
        scores = [score for _, _, score in utterance_difficulties]
        print(f"\n📊 Curriculum Order Statistics:")
        print(f"   Total utterances: {len(scores)}")
        print(f"   Easiest score: {min(scores):.4f}")
        print(f"   Hardest score: {max(scores):.4f}")
        print(f"   Mean score: {np.mean(scores):.4f}")
        print(f"   Median score: {np.median(scores):.4f}")
    
    def get_curriculum_subset_indices(self, competence):
        """
        Get indices for curriculum learning based on current competence level.
        
        Args:
            competence: Float in [0, 1] indicating training progress
                       0 = start (easiest samples only)
                       1 = end (all samples)
        
        Returns:
            List of dataset indices to use for this competence level
        """
        if self.curriculum_order is None:
            # No curriculum, return all indices
            return list(range(len(self)))
        
        # Calculate how many samples to include
        total_samples = len(self.curriculum_order)
        num_samples = max(1, int(competence * total_samples))
        
        # Return the easiest num_samples
        return self.curriculum_order[:num_samples]
    
    def _load_audio(self, wav_path):
        """
        Load audio waveform from path.
        
        Args:
            wav_path: Path to .wav file
        
        Returns:
            waveform: [time] tensor (mono)
            sample_rate: int
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Remove channel dimension: [1, time] -> [time]
        waveform = waveform.squeeze(0)
        
        return waveform, sample_rate
    
    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.utterance_ids)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        For training:
            Returns ensemble of teacher logits for the same utterance (matched by core-ID)
            
        For testing:
            Returns logits from the held-out speaker
        
        Returns:
            dict containing:
                - 'teacher_logits': Ensemble of teacher logits [num_teachers, max_seq_len, vocab_size]
                - 'num_teachers': Number of teachers for this utterance
                - 'teacher_speakers': List of teacher speaker IDs
                - 'core_id': Core utterance ID (without speaker prefix)
                - 'lengths': Length of each teacher's logit sequence
                - 'target_text': Ground truth text (if available in metadata)
        """
        if self.split == "train":
            # Get core-ID for this sample
            core_id = self.utterance_ids[idx]
            teacher_dict = self.coreid_to_speakers[core_id]
            
            # Get audio from first available teacher (they're the same utterance)
            first_speaker = list(teacher_dict.keys())[0]
            first_local_idx = teacher_dict[first_speaker]
            
            # Get full utterance ID from metadata
            first_metadata = self.metadata[first_speaker]
            utt_ids = first_metadata.get('utterance_ids') or first_metadata.get('utterances', [])
            full_utt_id = utt_ids[first_local_idx] if first_local_idx < len(utt_ids) else f"{first_speaker}_{core_id}"
            
            # Load audio using CSV mapping
            wav_path = self.audio_paths[first_speaker].get(full_utt_id)
            if wav_path is None:
                raise KeyError(f"Audio path not found for {first_speaker}/{full_utt_id}")
            
            audio, sample_rate = self._load_audio(wav_path)
            
            # Collect logits from all available teachers for this core-ID
            teacher_logits_list = []
            teacher_speakers_list = []
            lengths_list = []
            target_text = None
            
            for speaker, local_idx in teacher_dict.items():
                logits = self.teacher_logits[speaker][local_idx]
                teacher_logits_list.append(logits)
                teacher_speakers_list.append(speaker)
                lengths_list.append(len(logits))
                
                # Get target text from first available teacher
                if target_text is None:
                    metadata = self.metadata[speaker]
                    targets = metadata.get('targets') or metadata.get('target_words', [])
                    if local_idx < len(targets):
                        target_text = targets[local_idx]
            
            # Pad to max length in this ensemble
            max_len = max(lengths_list)
            vocab_size = teacher_logits_list[0].shape[1]
            num_teachers = len(teacher_logits_list)
            
            # Stack and pad teacher logits [num_teachers, max_len, vocab_size]
            padded_logits = torch.zeros(num_teachers, max_len, vocab_size)
            for i, logits in enumerate(teacher_logits_list):
                seq_len = logits.shape[0]
                padded_logits[i, :seq_len, :] = logits
            
            return {
                'audio': audio,  # [time] - NEW!
                'sample_rate': sample_rate,  # int - NEW!
                'teacher_logits': padded_logits,  # [num_teachers, max_len, vocab_size]
                'num_teachers': num_teachers,
                'teacher_speakers': teacher_speakers_list,
                'lengths': torch.tensor(lengths_list),
                'core_id': core_id,
                'target_text': target_text,
                'wav_path': wav_path  # For debugging
            }
        
        else:  # test
            speaker, utt_id = self.utterance_ids[idx]
            logits = self.teacher_logits[speaker][utt_id]
            metadata = self.metadata[speaker]
            
            # Get utterance ID and target text
            utt_ids = metadata.get('utterance_ids') or metadata.get('utterances', [])
            targets = metadata.get('targets') or metadata.get('target_words', [])
            
            full_utt_id = utt_ids[utt_id] if utt_id < len(utt_ids) else f"{speaker}_{utt_id}"
            target_text = targets[utt_id] if utt_id < len(targets) else None
            
            # Load audio using CSV mapping
            wav_path = self.audio_paths[speaker].get(full_utt_id)
            if wav_path is None:
                raise KeyError(f"Audio path not found for {speaker}/{full_utt_id}")
            
            audio, sample_rate = self._load_audio(wav_path)
            
            return {
                'audio': audio,  # [time] - NEW!
                'sample_rate': sample_rate,  # int - NEW!
                'logits': logits,  # [seq_len, vocab_size]
                'speaker': speaker,
                'utterance_id': full_utt_id,
                'length': len(logits),
                'target_text': target_text,
                'wav_path': wav_path  # For debugging
            }


def collate_logits(batch):
    """
    Custom collate function for batching ensemble logits with variable lengths.
    
    For training batches: each item has teacher_logits [num_teachers, seq_len, vocab_size]
    For test batches: each item has logits [seq_len, vocab_size]
    
    Pads to the maximum length in the batch.
    """
    # Check if this is a training batch (has 'teacher_logits') or test batch (has 'logits')
    is_train = 'teacher_logits' in batch[0]
    
    if is_train:
        # Training batch: handle ensemble of teachers + audio
        batch_size = len(batch)
        max_teachers = max(item['num_teachers'] for item in batch)
        max_len = max(item['teacher_logits'].shape[1] for item in batch)
        vocab_size = batch[0]['teacher_logits'].shape[2]
        max_audio_len = max(item['audio'].shape[0] for item in batch)
        
        # Initialize padded tensors
        padded_teacher_logits = torch.zeros(batch_size, max_teachers, max_len, vocab_size)
        padded_audio = torch.zeros(batch_size, max_audio_len)
        audio_lengths = torch.zeros(batch_size, dtype=torch.long)
        num_teachers_list = []
        teacher_speakers_list = []
        lengths_list = []
        core_ids = []
        target_texts = []
        sample_rates = []
        wav_paths = []
        
        for i, item in enumerate(batch):
            num_teachers = item['num_teachers']
            seq_len = item['teacher_logits'].shape[1]
            audio_len = item['audio'].shape[0]
            
            # Copy teacher logits (pad teachers dimension if needed)
            padded_teacher_logits[i, :num_teachers, :seq_len, :] = item['teacher_logits']
            
            # Copy audio (pad time dimension)
            padded_audio[i, :audio_len] = item['audio']
            audio_lengths[i] = audio_len
            
            num_teachers_list.append(num_teachers)
            teacher_speakers_list.append(item['teacher_speakers'])
            lengths_list.append(item['lengths'])
            core_ids.append(item['core_id'])
            target_texts.append(item['target_text'])
            sample_rates.append(item['sample_rate'])
            wav_paths.append(item['wav_path'])
        
        return {
            'audio': padded_audio,  # [batch, max_audio_len] - NEW!
            'audio_lengths': audio_lengths,  # [batch] - NEW!
            'sample_rate': sample_rates[0],  # Assume all same - NEW!
            'teacher_logits': padded_teacher_logits,  # [batch, max_teachers, max_len, vocab_size]
            'num_teachers': torch.tensor(num_teachers_list),  # [batch]
            'teacher_speakers': teacher_speakers_list,  # List of lists
            'lengths': lengths_list,  # List of tensors
            'core_ids': core_ids,
            'target_texts': target_texts,
            'wav_paths': wav_paths  # For debugging
        }
    
    else:
        # Test batch: simple padding + audio
        max_len = max(item['logits'].shape[0] for item in batch)
        vocab_size = batch[0]['logits'].shape[1]
        batch_size = len(batch)
        max_audio_len = max(item['audio'].shape[0] for item in batch)
        
        padded_logits = torch.zeros(batch_size, max_len, vocab_size)
        padded_audio = torch.zeros(batch_size, max_audio_len)
        audio_lengths = torch.zeros(batch_size, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        speakers = []
        utterance_ids = []
        target_texts = []
        sample_rates = []
        wav_paths = []
        
        for i, item in enumerate(batch):
            seq_len = item['logits'].shape[0]
            audio_len = item['audio'].shape[0]
            
            padded_logits[i, :seq_len, :] = item['logits']
            padded_audio[i, :audio_len] = item['audio']
            audio_lengths[i] = audio_len
            lengths[i] = seq_len
            speakers.append(item['speaker'])
            utterance_ids.append(item['utterance_id'])
            target_texts.append(item.get('target_text'))
            sample_rates.append(item['sample_rate'])
            wav_paths.append(item['wav_path'])
        
        return {
            'audio': padded_audio,  # [batch, max_audio_len] - NEW!
            'audio_lengths': audio_lengths,  # [batch] - NEW!
            'sample_rate': sample_rates[0],  # Assume all same - NEW!
            'logits': padded_logits,  # [batch, max_len, vocab_size]
            'lengths': lengths,  # [batch]
            'speakers': speakers,
            'utterance_ids': utterance_ids,
            'target_texts': target_texts,
            'wav_paths': wav_paths  # For debugging
        }


if __name__ == "__main__":
    """Quick test of the dataset."""
    print("🧪 Testing LogitEnsembleDataset...")
    
    # Test HIGH intelligibility with M08 held out
    train_dataset = LogitEnsembleDataset(
        held_out_speaker="M08",
        split="train",
        intelligibility_level="HIGH"
    )
    
    print(f"\n✅ Train dataset size: {len(train_dataset)}")
    
    # Test loading a sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\n📦 Sample data:")
        print(f"   Logits shape: {sample['logits'].shape}")
        print(f"   Speaker: {sample['speaker']}")
        print(f"   Utterance ID: {sample['utterance_id']}")
    
    # Test test dataset
    test_dataset = LogitEnsembleDataset(
        held_out_speaker="M08",
        split="test",
        intelligibility_level="HIGH"
    )
    
    print(f"\n✅ Test dataset size: {len(test_dataset)}")
    
    print("\n🎉 Dataset test complete!")
