#!/bin/bash
set -euxo pipefail
for i in {1..4}; do
	for speaker in F0{2..5} M0{1,4,5} M{07..12} M14 M16; do
		CUDA_VISIBLE_DEVICES=1 python3 train.py hparams/exp/ua_SA_val_common.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=${i} --speaker=${speaker}
	done
done
