#!/bin/bash
set -euxo pipefail
# note - we also run nf here because D is shorter than E
CUDA_VISIBLE_DEVICES=2 python3 train-metric.py hparams/exp/ua_SA_skip_control_val_common_WRA.yaml --freeze_encoder_layers=3 --freeze_decoder_layers=0 --speaker=M14
CUDA_VISIBLE_DEVICES=2 python3 train-metric.py hparams/exp/ua_SA_skip_control_val_common_WRA.yaml --freeze_encoder_layers=3 --freeze_decoder_layers=0 --speaker=M16
#for i in {1..12}; do
for i in {4..12}; do
	for speaker in F0{2..5} M0{1,4,5} M{07..12} M14 M16; do
		CUDA_VISIBLE_DEVICES=2 python3 train-metric.py hparams/exp/ua_SA_skip_control_val_common_WRA.yaml --freeze_encoder_layers=${i} --freeze_decoder_layers=0 --speaker=${speaker}
	done
done
