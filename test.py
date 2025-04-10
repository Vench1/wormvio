import argparse
import os
import torch
import logging
from path import Path
from utils import custom_transform
from wormvio import WormVIO
from collections import defaultdict
from utils.kitti_eval import KITTI_tester
import numpy as np
import math

from config import get_config, print_usage


def test(opt):
    assert opt.model is not None
    # Create Dir
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
        # f"lr@w{opt.lr_warmup:.0e}j{opt.lr_joint:.0e}f{opt.lr_fine:.0e}"
    )
    formatted_time = version_name + mode_name + formatted_time
    file_dir = experiment_dir.joinpath('{}/'.format(formatted_time))
    file_dir.mkdir_p()
    result_dir = file_dir.joinpath('files/')
    result_dir.mkdir_p()

    # Model initialization
    model = WormVIO(opt)

    # GPU or CPU selections
    if opt.use_multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Initialize the tester
    tester = KITTI_tester(opt)

    model.load_state_dict(torch.load(opt.model))
    print('load model %s' % opt.model)

    model = model.to(opt.device)
    model.eval()

    errors = tester.eval(model, False, num_gpu=torch.cuda.device_count() if opt.use_multi_gpu else 1)
    tester.generate_plots(result_dir, 10)
    # tester.save_text(result_dir)

    for i, seq in enumerate(opt.val_seq):
        message = f"Seq: {seq}, t_rel: {tester.errors[i]['t_rel']:.4f}, r_rel: {tester.errors[i]['r_rel']:.4f}, "
        message += f"t_rmse: {tester.errors[i]['t_rmse']:.4f}, r_rmse: {tester.errors[i]['r_rmse']:.4f}, "
        message += f"d_usage: {tester.errors[i]['d_usage']:.4f}, s_usage: {tester.errors[i]['s_usage']:.4f}"
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
    # opt.model = "./best_5.10.pth"
    test(opt)
