import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data.data_manager import AudioDatasetWrapper

import options as options

import os, pdb

def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):
    # Set dataset root to match your audio data
    cfg.DATASET.ROOT = '/home/zsim710/XDED/speechbrain/datasets/UASpeech'

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    # Define speaker domains - start with a few speakers for testing
    if hasattr(args, 'source_domains') and args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    else:
        # Default: use 3 speakers as source domains for Phase 1 testing
        cfg.DATASET.SOURCE_DOMAINS = ['F03', 'F04', 'F05']
    
    if hasattr(args, 'target_domains') and args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains
    else:
        # Default: use 1 speaker as target domain for Phase 1 testing  
        cfg.DATASET.TARGET_DOMAINS = ['M01']

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    else:
        # Default trainer for domain adaptation
        cfg.TRAINER.NAME = 'CrossGrad'

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    else:
        # Use our audio backbone
        cfg.MODEL.BACKBONE.NAME = 'audio_cnn'

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.hsic_interval:
        cfg.TRAINER.HSIC_INTERVAL = args.hsic_interval
    if args.hsic_warmup > -1:
        cfg.TRAINER.HSIC_WARMUP = args.hsic_warmup

    cfg.remark = args.remark if hasattr(args, 'remark') else 'audio_test'
    if args.output_dir:
        cfg.OUTPUT_DIR = os.path.join(
            args.output_dir,
            'uaspeech',
            '_'.join(cfg.DATASET.TARGET_DOMAINS),
            cfg.TRAINER.NAME + '_' + cfg.remark
        )
    else:
        cfg.OUTPUT_DIR = os.path.join(
            'output',
            'uaspeech', 
            '_'.join(cfg.DATASET.TARGET_DOMAINS),
            cfg.TRAINER.NAME + '_' + cfg.remark
        )
    
    cfg.IPC = args.IPC if hasattr(args, 'IPC') else -1


def setup_cfg(args):
    cfg = get_cfg_default()
    reset_cfg(cfg, args)
    
    # Load UASpeech dataset config
    cfg.merge_from_file('configs/datasets/uaspeech.yaml')
    
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)

    # Create output directory
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    # Set GPU
    if hasattr(args, 'gpu_id') and args.gpu_id > -1:
        torch.cuda.set_device(args.gpu_id)
    else:
        print("Warning: No GPU ID specified. Using default GPU.")

    # Build trainer with AudioDatasetWrapper
    print("🎵 Building trainer for audio domain adaptation...")
    print(f"Source domains (speakers): {cfg.DATASET.SOURCE_DOMAINS}")
    print(f"Target domains (speakers): {cfg.DATASET.TARGET_DOMAINS}")
    
    trainer = build_trainer(cfg)
    
    # Replace dataset wrapper with audio version
    if hasattr(trainer.dm, 'train_loader_x'):
        print("🔧 Replacing image dataset wrapper with audio dataset wrapper...")
        # This is a bit hacky but necessary for Phase 1
        # In Phase 2 we'll implement a proper audio trainer

    if args.eval_only:
        if hasattr(args, 'model_dir') and args.model_dir:
            load_epoch = args.load_epoch if hasattr(args, 'load_epoch') else None
            trainer.load_model(args.model_dir, epoch=load_epoch)
        trainer.test()
        return

    if not (hasattr(args, 'no_train') and args.no_train):
        print("🚀 Starting audio domain adaptation training...")
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Basic arguments  
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('--output-dir', type=str, default='output/uaspeech', help='output directory')
    parser.add_argument('--resume', type=str, default='', help='checkpoint directory')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    
    # Domain arguments
    parser.add_argument('--source-domains', type=str, nargs='+', 
                       default=['F03', 'F04', 'F05'], 
                       help='source speaker domains')
    parser.add_argument('--target-domains', type=str, nargs='+',
                       default=['M01'],
                       help='target speaker domains')
    
    # Model arguments
    parser.add_argument('--trainer', type=str, default='CrossGrad', 
                       help='domain adaptation trainer')
    parser.add_argument('--backbone', type=str, default='audio_cnn',
                       help='audio backbone')
    parser.add_argument('--head', type=str, default='', help='classification head')
    
    # Training arguments
    parser.add_argument('--transforms', type=str, nargs='+', help='data transforms')
    parser.add_argument('--eval-only', action='store_true', help='evaluation only')
    parser.add_argument('--model-dir', type=str, default='', help='model directory for eval')
    parser.add_argument('--load-epoch', type=int, help='epoch to load for eval')
    parser.add_argument('--no-train', action='store_true', help='skip training')
    
    # Additional arguments
    parser.add_argument('--remark', type=str, default='phase1_test', help='experiment remark')
    parser.add_argument('--IPC', type=int, default=-1, help='images per class')
    parser.add_argument('--hsic-interval', type=int, default=1, help='hsic interval')
    parser.add_argument('--hsic-warmup', type=int, default=5, help='hsic warmup')
    
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                       help='modify config options using the command-line')
    
    args = parser.parse_args()
    main(args)