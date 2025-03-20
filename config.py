import argparse

def str2bool(v):
    return v.lower() in ("true", "1")

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument(
    '--imu_dropout', type=float, default=0, help=""
    "dropout for the IMU encoder. Default: 0")
net_arg.add_argument(
    '--v_f_len', type=int, default=512, help=""
    "visual feature length.")
net_arg.add_argument(
    '--i_f_len', type=int, default=256, help=""
    "imu feature length.")
net_arg.add_argument(
    '--imu_len', type=int, default=10, help=""
    "number of imu samples contained in each time step. Default: 10")
net_arg.add_argument(
    '--rnn_hidden_size', type=int, default=1024, help=""
    "size of the LSTM latent")
net_arg.add_argument(
    '--rnn_dropout_out', type=float, default=0.2, help=""
    "dropout for the LSTM output layer")
net_arg.add_argument(
    '--rnn_dropout_between', type=float, default=0.2, help=""
    "dropout within LSTM")
net_arg.add_argument(
    "--fuse_method", type=str, default='ncp_IB', help=""
    "method of feature fusion. Options: 'cat', 'soft', 'hard', 'rnn_IB', 'ncp_IB'.")
net_arg.add_argument(
    '--ncp_units', type=int, default=32, help=""
    "units number of CfC/LTC ncps. Default: 32")
# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    '--seq_len', type=int, default=11, help=""
    "sequence length")
data_arg.add_argument(
    '--img_w', type=int, default=512, help=""
    "image width")
data_arg.add_argument(
    '--img_h', type=int, default=256, help=""
    "image height")
data_arg.add_argument(
    '--hflip', type=str2bool, default=True, help=""
    "whether to use horizonal flipping as augmentation")
data_arg.add_argument(
    '--color', type=str2bool, default=True, help=""
    "whether to use color augmentations")
data_arg.add_argument(
    "--data_dir", type=str, default='/home/b311/data4/zhaojiacheng/datasets/KITTI/kitti_odometry_color', help=""
    "path to the dataset")
data_arg.add_argument(
    '--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help=""
    "sequences for training")
data_arg.add_argument(
    '--val_seq', type=list, default=['05', '07', '10'], help=""
    # '--val_seq', type=list, default=['07'], help=""
    "sequences for validation/test")
data_arg.add_argument(
    '--seed', type=int, default=0, help=""
    "random seed. default=0")
# degradation mode
# 0: normal data 1: occlusion 2: blur 3: image missing 4: imu noise and bias 5: imu missing
# 6: spatial misalignment 7: temporal misalignment 8: vision degradation 9: all degradation
# 10: inertial degradation
data_arg.add_argument(
    '--train_degradation_mode', type=int, default=9, help=""
    "degradation mode of train dataset")
data_arg.add_argument(
    '--test_degradation_mode', type=int, default=9, help=""
    "degradation mode of test dataset")
# -----------------------------------------------------------------------------
# Loss
loss_arg = add_argument_group("loss")
loss_arg.add_argument(
    "--momentum", type=float, default=0.9, help=""
    "momentum")
loss_arg.add_argument(
    '--optimizer', type=str, default='Adam', help=""
    "type of optimizer [Adam, SGD]")
loss_arg.add_argument(
    '--weight_decay', type=float, default=5e-6, help=""
    "weight decay for the optimizer. default=5e-6")
loss_arg.add_argument(
    '--weighted', type=str2bool, default=False, help=""
    "whether to use weighted sum. default=False")
loss_arg.add_argument(
    '--Lambda', type=float, default=3e-5, help=""
    "penalty factor for the fusion")
loss_arg.add_argument(
    '--Gamma', type=float, default=9e-5, help=""
    "penalty factor for the visual selection")
# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument(
    "--batch_size", type=int, default=16, help=""
    "batch size")
train_arg.add_argument(
    "--workers", type=int, default=32, help=""
    "number of workers")
train_arg.add_argument(
    '--epochs_warmup', type=int, default=40, help=""
    "number of epochs for warmup. default=40")
train_arg.add_argument(
    '--epochs_joint', type=int, default=40, help=""
    "number of epochs for joint training. default=40")
train_arg.add_argument(
    '--epochs_fine', type=int, default=20, help=""
    "number of epochs for finetuning. default=20")
train_arg.add_argument(
    '--lr_warmup', type=float, default=5e-4, help=""
    "learning rate for warming up stage. default:5e-4")
train_arg.add_argument(
    '--lr_joint', type=float, default=5e-5, help=""
    "learning rate for joint training stage. default=5e-5")
train_arg.add_argument(
    '--lr_fine', type=float, default=1e-6, help=""
    "learning rate for finetuning stage. default=1e-6")
train_arg.add_argument(
    '--temp_init', type=int, default=5, help=""
    "initial temperature for gumbel-softmax. default=5")
train_arg.add_argument(
    '--eta', type=float, default=0.05, help=""
    "exponential decay factor for temperature. default=0.05")
train_arg.add_argument(
    '--device', type=str, default='cuda', help=""
    "whether to use multi gpu.")
train_arg.add_argument(
    '--use_multi_gpu', type=str2bool, default=False, help=""
    "whether to use multi gpu. default=False")
train_arg.add_argument(
    '--pretrain_flownet', type=str, default="./flownets_bn_EPE2.459.pth.tar", help=""
    "path to the pretrained flownet model")
train_arg.add_argument(
    '--pretrain', type=str, default=None, help=""
    "path to the pretrained model")
#------------------------------------------------------------------------------
# Logs
log = add_argument_group('Logs')
log.add_argument(
    '--save_dir', type=str, default='./results', help=""
    "path to save the result")
log.add_argument(
    '--experiment_name', type=str, default='',
    help='experiment name')
log.add_argument(
    '--print_frequency', type=int, default=180, help=""
    "print frequency for loss values")



def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
