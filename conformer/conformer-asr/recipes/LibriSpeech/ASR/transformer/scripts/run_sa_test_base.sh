#!/bin/bash
set -euxo pipefail

for speaker in F0{2..5} M0{1,4,5} M{07..12} M14 M16; do
	CUDA_VISIBLE_DEVICES=1 python3 train-metric.py hparams/exp/ua_sa_test_base_WRA_wo_finetuning.yaml --speaker=${speaker}
done
