for i in {0..4}; do
       CUDA_VISIBLE_DEVICES=1 python3 train.py hparams/exp/torgo/torgo_control_template.yaml --freeze_encoder_layers=0 --freeze_decoder_layers=${i}
done
