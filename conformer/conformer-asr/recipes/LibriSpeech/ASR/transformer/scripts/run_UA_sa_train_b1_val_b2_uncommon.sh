#!/bin/bash
set -euxo pipefail
for i in {0..4}; do
	for speaker in F0{2..5} M0{1,4,5} M{07..12} M14 M16; do
		CUDA_VISIBLE_DEVICES=2 python3 train.py hparams/exp/uaspeech/ua_SA_train_B1_val_B2_uncommon_WRA.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=${i} --speaker=${speaker}
	done
done
for i in {0..12}; do
	for speaker in F0{2..5} M0{1,4,5} M{07..12} M14 M16; do
		CUDA_VISIBLE_DEVICES=2 python3 train.py hparams/exp/uaspeech/ua_SA_train_B1_val_B2_uncommon_WRA.yaml --freeze_decoder_layers=0 --freeze_encoder_layers=${i} --speaker=${speaker}
	done
done
