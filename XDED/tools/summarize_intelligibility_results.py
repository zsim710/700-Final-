#!/usr/bin/env python3
"""Summarize intelligibility-based averaging results (HIGH and VERY_LOW bands)."""

import json
import os
from pathlib import Path

def summarize_results():
    result_dir = "/home/zsim710/XDED/XDED/results/intelligibility_averaging/evaluation"
    
    scenarios = ['excluded', 'included']
    bands = ['HIGH', 'VERY_LOW']
    test_speakers = {'HIGH': 'M08', 'VERY_LOW': 'M01'}
    
    print("="*80)
    print("INTELLIGIBILITY-BASED AVERAGING RESULTS SUMMARY")
    print("Bands: HIGH and VERY_LOW")
    print("="*80)
    
    for scenario in scenarios:
        print(f"\n{scenario.upper()} SCENARIO (Test speaker {scenario} from averaging):")
        print("-"*80)
        print(f"{'Band':<15} {'Test Speaker':<15} {'WER (%)':<12} {'Accuracy (%)':<15}")
        print("-"*80)
        
        for band in bands:
            speaker = test_speakers[band]
            result_file = os.path.join(result_dir, scenario, f"{band}_{speaker}_results.json")
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    results = json.load(f)
                    wer = results.get('wer', 'N/A')
                    acc = results.get('accuracy', 'N/A')
                    
                    if isinstance(wer, (int, float)):
                        wer_str = f"{wer:.2f}"
                    else:
                        wer_str = str(wer)
                    
                    if isinstance(acc, (int, float)):
                        acc_str = f"{acc:.2f}"
                    else:
                        acc_str = str(acc)
                    
                    print(f"{band:<15} {speaker:<15} {wer_str:<12} {acc_str:<15}")
            else:
                print(f"{band:<15} {speaker:<15} {'N/A':<12} {'N/A':<15}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("EXCLUDED: Test speaker NOT included in weight averaging (true generalization)")
    print("INCLUDED: Test speaker included in weight averaging (should perform better)")
    print("\nExpected: INCLUDED scenario should show lower WER than EXCLUDED")
    print("="*80)

if __name__ == "__main__":
    summarize_results()
