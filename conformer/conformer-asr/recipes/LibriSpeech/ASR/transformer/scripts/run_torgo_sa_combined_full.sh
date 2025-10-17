for i in {0..4}; do
	for speaker in F{01,03,04} M{01..05}; do
		CUDA_VISIBLE_DEVICES=2 python3 train.py hparams/exp/torgo_SA_fd1_WER_combined.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=${i} --speaker=${speaker}
		#CUDA_VISIBLE_DEVICES=2 python3 test.py hparams/exp/torgo_SA_fd1_WER.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=${i} --speaker=${speaker}
	done
done
for i in {1..12}; do
	for speaker in F{01,03,04} M{01..05}; do
		CUDA_VISIBLE_DEVICES=2 python3 train.py hparams/exp/torgo_SA_fd1_WER_combined.yaml --freeze_encoder_layers=${i} --freeze_decoder_layers=0 --speaker=${speaker}
		#CUDA_VISIBLE_DEVICES=2 python3 test.py hparams/exp/torgo_SA_fd1_WER.yaml --freeze_encoder_layers=${i} --freeze_decoder_layers=0 --speaker=${speaker}
	done
done
