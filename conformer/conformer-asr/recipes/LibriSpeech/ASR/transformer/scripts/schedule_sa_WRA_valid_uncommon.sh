#!/bin/bash
set -euxo pipefail

# wait until gpu is free 
while true; do
	gpu_utilisation=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1)
	if [ "$gpu_utilisation" -eq 0 ]; then
		echo "GPU 1 is free now."
		break
	else
		echo "busy: $gpu_utilisation"
		sleep 60
	fi
done

# note - we also run nf here because D is shorter than E
#for i in {0..4}; do
#	for speaker in F0{2..5} M0{1,4,5} M{07..12} M14 M16; do
#		CUDA_VISIBLE_DEVICES=1 python3 train-metric.py hparams/exp/ua_SA_val_uncommon_WRA.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=${i} --speaker=${speaker}
#	done
#done
for speaker in M{11..12} M14 M16; do
	CUDA_VISIBLE_DEVICES=1 python3 train-metric.py hparams/exp/ua_SA_val_uncommon_WRA.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=3 --speaker=${speaker}
done
for speaker in F0{2..5} M0{1,4,5} M{07..12} M14 M16; do
	CUDA_VISIBLE_DEVICES=1 python3 train-metric.py hparams/exp/ua_SA_val_uncommon_WRA.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=4 --speaker=${speaker}
done

#for i in {1..12}; do
#	for speaker in F0{2..5} M0{1,4,5} M{07..12} M14 M16; do
#		CUDA_VISIBLE_DEVICES=1 python3 train-metric.py hparams/exp/ua_SA_val_uncommon_WRA.yaml --freeze_encoder_layers=${i} --freeze_decoder_layers=0 --speaker=${speaker}
#	done
#done
