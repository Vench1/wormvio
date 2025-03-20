import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from wormvio import WormVIO
from collections import defaultdict
from utils.kitti_eval import KITTI_tester
import numpy as np
import math

from config import get_config, print_usage


def create_dir(opt):
    experiment_dir = Path(opt.save_dir)
    experiment_dir.mkdir_p()
    if opt.experiment_name == "":
        version_name = ""
    else:
        version_name = opt.experiment_name + "_"
    if opt.fuse_method == "ncp_IB":
        mode_name = f"ncp{opt.ncp_units}_"
    elif opt.fuse_method == "rnn_IB":
        mode_name = "rnn64_"
    else:
        mode_name = f"{opt.fuse_method}_"
    formatted_time = (
        f"bz@{opt.batch_size}_lambda@{opt.Lambda:.1e}_Gamma@{opt.Gamma:.1e}"
        # f"_DM@{opt.train_degradation_mode}-{opt.test_degradation_mode}"
        # f"_lr@w{opt.lr_warmup:.0e}j{opt.lr_joint:.0e}f{opt.lr_fine:.0e}"
    )
    formatted_time = version_name + mode_name + formatted_time
    file_dir = experiment_dir.joinpath('{}/'.format(formatted_time))
    file_dir.mkdir_p()
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir_p()
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir_p()
    return checkpoints_dir, log_dir


def update_status(ep, opt, model):
    if ep < opt.epochs_warmup:  # Warmup stage
        lr = opt.lr_warmup
        temp = opt.temp_init
        random = True
        for param in model.IB.parameters():
            param.requires_grad = False
        msg = "Warmup modules now."
    elif opt.epochs_warmup <= ep < opt.epochs_warmup + opt.epochs_joint:  # Joint training stage
        lr = opt.lr_joint
        temp = opt.temp_init * math.exp(-opt.eta * (ep - opt.epochs_warmup))
        random = False
        for param in model.IB.parameters():
            param.requires_grad = True
        msg = "Joint training now."
    else:  # Finetuning stage
        lr = opt.lr_fine
        temp = opt.temp_init * math.exp(-opt.eta * (ep - opt.epochs_warmup))
        random = False
        msg = "Fine-tune module now."
    return lr, msg, temp, random


def train_one_epoch(model, temp, random, optimizer, train_loader, logger, ep, device, weighted=False):
    pose_losses = []
    penalty_losses = []
    data_len = len(train_loader)

    for i, (imgs, imus, gts, rot, weight) in enumerate(train_loader):

        imgs = imgs.to(device).float()
        imus = imus.to(device).float()
        gts = gts.to(device).float()
        weight = weight.to(device).float()

        optimizer.zero_grad()

        poses, _, _, decisions, selections = model(imgs, imus, hc=None, ncp_hc=None, temp=temp, random=random)
        if not weighted:
            angle_loss = torch.nn.functional.mse_loss(poses[:, :, :3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(poses[:, :, 3:], gts[:, :, 3:])
        else:
            weight = weight / weight.sum()
            angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:, :, :3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:, :, 3:] - gts[:, :, 3:]) ** 2).mean()

        pose_loss = 100 * angle_loss + translation_loss

        penalty_d = (decisions[:, :, 0].float()).sum(-1).mean()
        penalty_s = (selections[:, :, 0].float()).sum(-1).mean()
        loss = pose_loss + penalty_d * opt.Lambda + penalty_s * opt.Gamma

        loss.backward()

        optimizer.step()

        if i % opt.print_frequency == 0:
            message = f'Epoch: {ep}, iters: {i}/{data_len}, pose_loss: {pose_loss.item():.6f}, penalty_d: {penalty_d.item():.6f}, penalty_s: {penalty_s.item():.6f}'
            print(message)
            logger.info(message)

        pose_losses.append(pose_loss.item())
        penalty_losses.append(penalty_d.item() + penalty_s.item())

    return np.mean(pose_losses), np.mean(penalty_losses)


def train_eval(opt):
    checkpoints_dir, log_dir = create_dir(opt)

    # Create logs
    logger = logging.getLogger(opt.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s.txt' % opt.experiment_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(opt)
    writer = SummaryWriter(log_dir=log_dir)

    # Load the dataset
    # transform_train = [custom_transform.ToTensor(),
    #                    custom_transform.Resize((opt.img_h, opt.img_w))]
    transform_train = [custom_transform.ToTensor()]
    if opt.hflip:
        transform_train += [custom_transform.RandomHorizontalFlip()]
    if opt.color:
        transform_train += [custom_transform.RandomColorAug()]
    transform_train = custom_transform.Compose(transform_train)

    train_dataset = KITTI(opt.data_dir,
                          sequence_length=opt.seq_len,
                          train_seqs=opt.train_seq,
                          transform=transform_train,
                          data_degradation=opt.train_degradation_mode
                          )
    logger.info('train_dataset: ' + str(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )

    # Model initialization
    model = WormVIO(opt)

    # Continual training or not
    if opt.pretrain is not None:
        model.load_state_dict(torch.load(opt.pretrain))
        print('load model %s' % opt.pretrain)
        logger.info('load model %s' % opt.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    # Use the pre-trained flownet or not
    if opt.pretrain_flownet and opt.pretrain is None:
        pretrained_w = torch.load(opt.pretrain_flownet, map_location='cpu')
        model_dict = model.LS.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        model.LS.load_state_dict(model_dict)

    # GPU or CPU selections
    if opt.use_multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(opt.device)

    # Initialize the tester
    tester = KITTI_tester(opt)

    pretrain = opt.pretrain
    init_epoch = int(pretrain[-7:-4]) + 1 if opt.pretrain is not None else 0

    # Initialize the optimizer
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=opt.weight_decay)

    best = 10000

    for ep in range(init_epoch, opt.epochs_warmup + opt.epochs_joint + opt.epochs_fine):

        lr, msg, temp, random = update_status(ep, opt, model)
        optimizer.param_groups[0]['lr'] = lr
        message = f'Epoch: {ep}, lr: {lr} | {msg} random: {random}, temperaure: {temp:.5f}'
        print(message)
        logger.info(message)

        model.train()
        avg_pose_loss, avg_penalty_loss = train_one_epoch(model, temp, random, optimizer, train_loader, logger, ep,
                                                          opt.device,
                                                          weighted=opt.weighted)

        # Save the model after training
        ls = model.timer.avg('visual')
        md = model.timer.avg('inertial')
        pi = model.timer.avg('pose&IB')
        torch.save(model.state_dict(), f'{checkpoints_dir}/{ep:003}.pth')
        message = f'Epoch {ep} training finished, pose loss: {avg_pose_loss:.6f}, penalty loss: {avg_penalty_loss:.6f}, model saved'
        message += f'\ntime consume: ls={ls:.4f}s | md={md:.4f}s | pose & IB={pi:.4f}s'
        print(message)
        logger.info(message)
        writer.add_scalar('Loss/pose', avg_pose_loss, ep)
        writer.add_scalar('Loss/penalty', avg_penalty_loss, ep)

        if ep > opt.epochs_warmup + opt.epochs_joint:
        # if ep > opt.epochs_warmup:
            # Evaluate the model
            print('Evaluating the model')
            logger.info('Evaluating the model')
            with torch.no_grad():
                model.eval()
                errors = tester.eval(model, random=random,
                                     num_gpu=torch.cuda.device_count() if opt.use_multi_gpu else 1)

            t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
            r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
            t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
            r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])
            d_usage = np.mean([errors[i]['d_usage'] for i in range(len(errors))])
            s_usage = np.mean([errors[i]['s_usage'] for i in range(len(errors))])

            if t_rel < best:
                best = t_rel
                torch.save(model.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')

            message = f'Epoch {ep} evaluation finished , t_rel: {t_rel:.4f}, r_rel: {r_rel:.4f}, t_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f}, \nd_usage: {d_usage:.4f}, s_usage: {s_usage:.4f}, best t_rel: {best:.4f}'
            logger.info(message)
            print(message)

    message = f'Training finished, best t_rel: {best:.4f}'
    writer.close()
    logger.info(message)
    print(message)


if __name__ == "__main__":
    opt, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    # Set the random seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # opt.pretrain = "/results/o9ch_CfC32_bz@16_lambda@3e-05_Gamma@3e-05_DM@9-9/checkpoints/039.pth"

    train_eval(opt)
