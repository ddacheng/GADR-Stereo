import torch
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os

import nets
import dataloader
from dataloader import transforms
from utils import utils
import model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str,
                    help='Validation mode on small subset or test mode on full test data')

# Training data
parser.add_argument('--data_dir', default='data/KITTI', type=str, help='Training dataset')
parser.add_argument('--dataset_name', default='KITTI_mix', type=str, help='Dataset name')

parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--val_batch_size', default=4, type=int, help='Batch size for validation')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for data loading')
parser.add_argument('--img_height', default=320, type=int, help='Image height for training')
parser.add_argument('--img_width', default=960, type=int, help='Image width for training')

# For KITTI, using 384x1248 for validation
parser.add_argument('--val_img_height', default=384, type=int, help='Image height for validation')
parser.add_argument('--val_img_width', default=1248, type=int, help='Image width for validation')

# Model
parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')
parser.add_argument('--checkpoints_dir', default='checkpoints/newnet_bg_kit', type=str, help='Directory to save model checkpoints and logs')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
parser.add_argument('--max_epoch', default=1000, type=int, help='Maximum epoch number for training')
parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')

# AANet


parser.add_argument('--feature_similarity', default='correlation', type=str,
                    help='Similarity measure for matching cost')
parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction')

parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')

parser.add_argument('--num_stage_blocks', default=1, type=int, help='Number of deform blocks for ISA')

parser.add_argument('--no_intermediate_supervision', action='store_true',
                    help='Whether to add intermediate supervision')

parser.add_argument('--refinement_type', default='stereodrnet', help='Type of refinement module')

parser.add_argument('--pretrained_aanet', default='checkpoints/newnet_bg_sceneflow/aanet_best.pth', type=str, help='Pretrained network')#'checkpoints/aanet_sceneflow/aanet_best.pth'
parser.add_argument('--freeze_bn', action='store_true', help='Switch BN to eval mode to fix running statistics')

# Learning rate
parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help='Decay gamma')
parser.add_argument('--lr_scheduler_type', default='MultiStepLR', help='Type of learning rate scheduler')
parser.add_argument('--milestones', default='400,600,800,900', type=str, help='Milestones for MultiStepLR')

# Loss
parser.add_argument('--highest_loss_only', action='store_true', help='Only use loss on highest scale for finetuning')
parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')

# Log
parser.add_argument('--print_freq', default=100, type=int, help='Print frequency to screen (iterations)')
parser.add_argument('--summary_freq', default=100, type=int, help='Summary frequency to tensorboard (iterations)')
parser.add_argument('--no_build_summary', action='store_true', help='Dont save sammary when training to save space')
parser.add_argument('--save_ckpt_freq', default=100, type=int, help='Save checkpoint frequency (epochs)')

parser.add_argument('--evaluate_only', action='store_true', help='Evaluate pretrained models')
parser.add_argument('--no_validate', action='store_true', help='No validation')
parser.add_argument('--strict', action='store_true', help='Strict mode when loading checkpoints')
parser.add_argument('--val_metric', default='epe', help='Validation metric to select best model')

args = parser.parse_args()
logger = utils.get_logger()

utils.check_path(args.checkpoints_dir)
utils.save_args(args)

filename = 'command_test.txt' if args.mode == 'test' else 'command_train.txt'
utils.save_command(args.checkpoints_dir, filename)


def main():
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train loader
    train_transform_list = [transforms.RandomCrop(args.img_height, args.img_width),
                            transforms.RandomColor(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
    train_transform = transforms.Compose(train_transform_list)

    train_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                          dataset_name=args.dataset_name,
                                          mode='train' if args.mode != 'train_all' else 'train_all',
                                          load_pseudo_gt=args.load_pseudo_gt,
                                          transform=train_transform)

    logger.info('=> {} training samples found in the training set'.format(len(train_data)))

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Validation loader
    val_transform_list = [transforms.RandomCrop(args.val_img_height, args.val_img_width, validate=True),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                         ]
    val_transform = transforms.Compose(val_transform_list)
    val_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                        dataset_name=args.dataset_name,
                                        mode=args.mode,
                                        transform=val_transform)

    val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Network
    fastnet = nets.FastNet(args.max_disp,
                       num_downsample=args.num_downsample,
                       feature_similarity=args.feature_similarity,
                       num_scales=args.num_scales,
                       num_stage_blocks=args.num_stage_blocks,
                       no_intermediate_supervision=args.no_intermediate_supervision,
                       refinement_type=args.refinement_type).to(device)

    logger.info('%s' % fastnet)

    if args.pretrained_aanet is not None:
        logger.info('=> Loading pretrained AANet: %s' % args.pretrained_aanet)
        # Enable training from a partially pretrained model
        utils.load_pretrained_net(fastnet, args.pretrained_aanet, no_strict=(not args.strict))

    if torch.cuda.device_count() > 1:
        logger.info('=> Use %d GPUs' % torch.cuda.device_count())
        fastnet = torch.nn.DataParallel(fastnet)

    # Save parameters
    num_params = utils.count_parameters(fastnet)
    logger.info('=> Number of trainable parameters: %d' % num_params)
    save_name = '%d_parameters' % num_params
    open(os.path.join(args.checkpoints_dir, save_name), 'a').close()

    # Optimizer
    # Learning rate for offset learning is set 0.1 times those of existing layers
    specific_params = list(filter(utils.filter_specific_params,
                                  fastnet.named_parameters()))
    base_params = list(filter(utils.filter_base_params,
                              fastnet.named_parameters()))

    specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
    base_params = [kv[1] for kv in base_params]

    specific_lr = args.learning_rate * 0.1
    params_group = [
        {'params': base_params, 'lr': args.learning_rate},
        {'params': specific_params, 'lr': specific_lr},
    ]

    optimizer = torch.optim.Adam(params_group, weight_decay=args.weight_decay)

    # Resume training
    if args.resume:
        # AANet
        start_epoch, start_iter, best_epe, best_epoch = utils.resume_latest_ckpt(
            args.checkpoints_dir, fastnet, 'fastnet')

        # Optimizer
        utils.resume_latest_ckpt(args.checkpoints_dir, optimizer, 'optimizer')
    else:
        start_epoch = 0
        start_iter = 0
        best_epe = None
        best_epoch = None

    # LR scheduler
    if args.lr_scheduler_type is not None:
        last_epoch = start_epoch if args.resume else start_epoch - 1
        if args.lr_scheduler_type == 'MultiStepLR':
            milestones = [int(step) for step in args.milestones.split(',')]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=milestones,
                                                                gamma=args.lr_decay_gamma,
                                                                last_epoch=last_epoch)
        else:
            raise NotImplementedError

    train_model = model.Model(args, logger, optimizer, fastnet, device, start_iter, start_epoch,
                              best_epe=best_epe, best_epoch=best_epoch)

    logger.info('=> Start training...')

    if args.evaluate_only:
        assert args.val_batch_size == 1
        train_model.validate(val_loader)
    else:
        for _ in range(start_epoch, args.max_epoch):
            if not args.evaluate_only:
                train_model.train(train_loader)
            if not args.no_validate:
                train_model.validate(val_loader)
            if args.lr_scheduler_type is not None:
                lr_scheduler.step()

        logger.info('=> End training\n\n')


if __name__ == '__main__':
    main()
