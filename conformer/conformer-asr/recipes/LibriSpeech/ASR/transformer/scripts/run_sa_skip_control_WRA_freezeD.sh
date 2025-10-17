#!/bin/bash
set -euxo pipefail
# note - we also run nf here because D is shorter than E
#for i in {0..4}; do
for i in {3..4}; do # recover from skip_control_UA_WRA_F03_D3E0
	for speaker in F0{3..5} M0{1,4,5} M{07..12} M14 M16; do
		CUDA_VISIBLE_DEVICES=1 python3 train-metric.py hparams/exp/ua_SA_skip_control_val_common_WRA.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=${i} --speaker=${speaker}
	done
done
CUDA_VISIBLE_DEVICES=1 python3 train-metric.py hparams/exp/ua_SA_skip_control_val_common_WRA.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=4 --speaker=F02
