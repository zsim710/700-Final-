for i in {1..12}; do
       CUDA_VISIBLE_DEVICES=2 python3 train.py hparams/exp/torgo_control_template.yaml --freeze_encoder_layers=${i} --freeze_decoder_layers=0
done
