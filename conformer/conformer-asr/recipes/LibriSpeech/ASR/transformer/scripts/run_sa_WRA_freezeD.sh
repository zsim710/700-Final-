#!/bin/bash
set -euxo pipefail
# note - we also run nf here because D is shorter than E
for i in {0..4}; do
	for speaker in F0{2..5} M0{1,4,5} M{07..12} M14 M16; do
		CUDA_VISIBLE_DEVICES=1 python3 train-metric.py hparams/exp/ua_SA_val_common_WRA.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=${i} --speaker=${speaker}
	done
done
