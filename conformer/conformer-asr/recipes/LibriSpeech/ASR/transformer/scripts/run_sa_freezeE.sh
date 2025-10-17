#!/bin/bash
set -euxo pipefail

function exp_freeze_encoder_layer() {
	i=$1
	for speaker in F0{2..5} M0{1,4,5} M{07..12} M14 M16; do
		CUDA_VISIBLE_DEVICES=2 python3 train.py hparams/exp/ua_SA_val_common.yaml --freeze_encoder_layers=${i} --freeze_decoder_layers=0 --speaker=${speaker}
	done
}

#sleep 14400 # wait nf's 15min*15speakers to finish
#for i in {1..12}; do
#	exp_freeze_encoder_layer $i
#done


# crashed at E10 M08
for speaker in M{08..12} M14 M16; do
	CUDA_VISIBLE_DEVICES=2 python3 train.py hparams/exp/ua_SA_val_common.yaml --freeze_encoder_layers=10 --freeze_decoder_layers=0 --speaker=${speaker}
done
for i in {11..12}; do
	exp_freeze_encoder_layer $i
done
