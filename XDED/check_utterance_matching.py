#!/usr/bin/env python3
"""
Check utterance matching across all speakers to determine best ensemble strategy.

This script:
1. Loads metadata for all 15 speakers
2. Extracts utterance IDs and target text
3. Analyzes overlap patterns
4. Recommends matching strategy
"""

import json
import os
from collections import defaultdict, Counter

logit_root = "/home/zsim710/XDED/speechbrain/exp_results/logit_extraction"

all_speakers = ["F02", "F03", "F04", "F05", "M01", "M04", "M05", 
                "M07", "M08", "M09", "M10", "M11", "M12", "M14", "M16"]

print("=" * 80)
print("🔍 UTTERANCE MATCHING ANALYSIS")
print("=" * 80)

# Step 1: Load all metadata
print("\n📊 Step 1: Loading metadata for all speakers...")
print("-" * 80)

speaker_metadata = {}
speaker_utterances = {}
speaker_targets = {}

for speaker in all_speakers:
    metadata_path = os.path.join(logit_root, speaker, speaker, f"{speaker}_metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        speaker_metadata[speaker] = metadata
        
        # Extract utterance IDs
        if 'utterance_ids' in metadata:
            speaker_utterances[speaker] = metadata['utterance_ids']
            print(f"✅ {speaker}: {len(metadata['utterance_ids'])} utterances")
        else:
            print(f"⚠️  {speaker}: No 'utterance_ids' field in metadata")
        
        # Extract targets (text)
        if 'target_words' in metadata:
            speaker_targets[speaker] = metadata['target_words']
        
        # Show sample data
        if speaker in ["F03", "M01", "M08"]:  # Representative samples
            print(f"   Sample utterance IDs:")
            for i in range(min(3, len(metadata.get('utterance_ids', [])))):
                utt_id = metadata['utterance_ids'][i]
                target = metadata.get('target_words', [None])[i] if 'target_words' in metadata else "N/A"
                print(f"      [{i}] ID: {utt_id}")
                print(f"          Text: {target}")
    else:
        print(f"❌ {speaker}: Metadata file not found")

# Step 2: Analyze utterance ID patterns
print("\n\n📋 Step 2: Analyzing utterance ID patterns...")
print("-" * 80)

# Check if utterance IDs contain speaker prefix
sample_speaker = "F03"
if sample_speaker in speaker_utterances:
    sample_ids = speaker_utterances[sample_speaker][:5]
    print(f"\nSample utterance IDs from {sample_speaker}:")
    for utt_id in sample_ids:
        print(f"  - {utt_id}")
    
    # Check if IDs are speaker-specific or shared format
    has_speaker_prefix = any(sample_speaker in utt_id for utt_id in sample_ids)
    print(f"\n{'✅' if has_speaker_prefix else '❌'} Utterance IDs contain speaker prefix")

# Step 3: Count utterances per speaker
print("\n\n📊 Step 3: Utterance counts per speaker...")
print("-" * 80)

utterance_counts = {spk: len(utts) for spk, utts in speaker_utterances.items()}
sorted_speakers = sorted(utterance_counts.items(), key=lambda x: x[1])

print(f"\n{'Speaker':<8} {'Count':<8} {'Bar'}")
print("-" * 50)
max_count = max(utterance_counts.values())
for speaker, count in sorted_speakers:
    bar = "█" * int(40 * count / max_count)
    print(f"{speaker:<8} {count:<8} {bar}")

# Identify outliers
mean_count = sum(utterance_counts.values()) / len(utterance_counts)
std_count = (sum((c - mean_count)**2 for c in utterance_counts.values()) / len(utterance_counts)) ** 0.5

print(f"\n📈 Statistics:")
print(f"   Mean: {mean_count:.0f} utterances")
print(f"   Std Dev: {std_count:.0f}")
print(f"   Min: {min(utterance_counts.values())} ({sorted_speakers[0][0]})")
print(f"   Max: {max(utterance_counts.values())} ({sorted_speakers[-1][0]})")

# Identify outliers (> 1.5 std below mean)
outliers = [spk for spk, count in utterance_counts.items() 
            if count < mean_count - 1.5 * std_count]
if outliers:
    print(f"\n⚠️  Outliers (significantly fewer utterances):")
    for spk in outliers:
        print(f"   - {spk}: {utterance_counts[spk]} utterances")
else:
    print(f"\n✅ No significant outliers detected")

# Step 4: Check for shared utterances using utterance IDs (normalized)
print("\n\n🔍 Step 4: Analyzing utterance ID overlap (normalize by removing speaker prefix)...")
print("-" * 80)

# Normalize utterance IDs by removing the leading speaker prefix before the first '_'
coreid_to_speakers = defaultdict(set)
coreid_to_ids = defaultdict(dict)

for speaker, utt_ids in speaker_utterances.items():
    for utt in utt_ids:
        if '_' in utt:
            core = utt.split('_', 1)[1]
        else:
            core = utt
        coreid_to_speakers[core].add(speaker)
        coreid_to_ids[core][speaker] = utt

# Distribution: how many speakers per core utterance
core_count_distribution = Counter(len(speakers) for speakers in coreid_to_speakers.values())

print(f"\nDistribution of ID coverage (how many speakers share each core utterance):")
print(f"{'# Speakers':<12} {'# CoreIDs':<10} {'Percentage'}")
print("-" * 50)
total_coreids = len(coreid_to_speakers)
for num_speakers in sorted(core_count_distribution.keys(), reverse=True):
    num_coreids = core_count_distribution[num_speakers]
    pct = 100 * num_coreids / total_coreids if total_coreids > 0 else 0
    bar = "█" * int(30 * num_speakers / max(1, len(all_speakers)))
    print(f"{num_speakers:<12} {num_coreids:<10} {pct:>5.1f}% {bar}")

# Calculate matching options based on core IDs
all_14_speakers = [s for s in all_speakers if s != "M08"]  # Exclude test speaker

# Option 1: Strict matching (core id present in all 14 teachers)
strict_match_coreids = [core for core, speakers in coreid_to_speakers.items()
                        if all(s in speakers for s in all_14_speakers)]

# Option 2: Partial matching (at least 12/14 teachers)
partial_match_coreids_12 = [core for core, speakers in coreid_to_speakers.items()
                           if sum(s in speakers for s in all_14_speakers) >= 12]

# Option 2b: Partial matching (at least 10/14 teachers)
partial_match_coreids_10 = [core for core, speakers in coreid_to_speakers.items()
                           if sum(s in speakers for s in all_14_speakers) >= 10]

print("\n\n📊 Step 5: Matching Strategy Options...")
print("-" * 80)

print(f"\n{'Strategy':<30} {'# Texts':<10} {'Avg Teachers'}")
print("-" * 60)
print(f"{'Option 1: Strict (14/14)':<30} {len(strict_match_coreids):<10} 14.0")
print(f"{'Option 2a: Partial (≥12/14)':<30} {len(partial_match_coreids_12):<10} ~13.0")
print(f"{'Option 2b: Partial (≥10/14)':<30} {len(partial_match_coreids_10):<10} ~11.5")
print(f"{'Option 3: All (no matching)':<30} {sum(utterance_counts.values()):<10} ~1.0")

# Show some example texts and their coverage
print("\n\n📝 Example core IDs and speaker coverage:")
print("-" * 80)
sample_coreids = list(coreid_to_speakers.items())[:10]
for core, speakers in sample_coreids:
    coverage = len([s for s in all_14_speakers if s in speakers])
    print(f"\n'{core}': {coverage}/14 teachers")
    missing = [s for s in all_14_speakers if s not in speakers]
    if missing:
        print(f"   Missing: {', '.join(missing)}")

# Step 6: Recommendation
print("\n\n🎯 RECOMMENDATION")
print("=" * 80)

if outliers and len(outliers) <= 2:
    print(f"\n⚠️  Pattern: {len(outliers)} outlier speaker(s) with significantly fewer utterances:")
    for spk in outliers:
        print(f"   - {spk}: {utterance_counts[spk]} utterances")
    print(f"\nSupervisor's advice: 'if it's just one speaker it'd be cleaner to exclude'")
    print(f"\n✅ RECOMMENDED STRATEGY:")
    print(f"   1. Exclude outlier(s): {', '.join(outliers)}")
    print(f"   2. Use remaining {15 - len(outliers) - 1} teachers for M08 fold")
    print(f"   3. This should give more consistent utterance counts")
else:
    print(f"\n✅ Pattern: Missing utterances scattered across speakers (no single outlier)")
    print(f"\nSupervisor's advice: 'some data is better than no data'")
    print(f"\n✅ RECOMMENDED STRATEGY: Option 2a or 2b (Partial Matching)")
    print(f"   - Option 2a: Use core IDs with ≥12/14 teachers: {len(partial_match_coreids_12)} texts")
    print(f"   - Option 2b: Use core IDs with ≥10/14 teachers: {len(partial_match_coreids_10)} texts")
    print(f"   - Average available teachers per text: high quality ensemble")

print("\n" + "=" * 80)
print("✅ Analysis complete!")
print("=" * 80)
