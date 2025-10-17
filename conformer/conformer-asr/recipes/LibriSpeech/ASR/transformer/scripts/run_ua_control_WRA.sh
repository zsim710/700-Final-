#!/bin/bash
set -euxo pipefail

CUDA_VISIBLE_DEVICES=1 python3 train-metric.py hparams/exp/ua_control_LOSO_WRA.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=0

for i in {2..4}; do
	CUDA_VISIBLE_DEVICES=1 python3 train-metric.py hparams/exp/ua_control_LOSO_WRA.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=${i}
done

#for i in {1..12}; do
#	CUDA_VISIBLE_DEVICES=1 python3 train-metric.py hparams/exp/ua_control_LOSO_WRA.yaml --freeze_encoder_layers=${i} --freeze_decoder_layers=0
#done

