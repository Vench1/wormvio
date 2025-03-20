import os
import glob
import numpy as np
import time
import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import math
from utils.utils import *
from tqdm import tqdm
import random
import cv2


class data_partition():
    def __init__(self, opt, folder, data_degradation=0):
        super(data_partition, self).__init__()
        self.opt = opt
        self.data_dir = opt.data_dir
        self.seq_len = opt.seq_len
        self.folder = folder
        self.data_degradation = data_degradation
        self.load_data()

    def load_data(self):
        image_dir = self.data_dir + '/sequences/'
        imu_dir = self.data_dir + '/imus/'
        pose_dir = self.data_dir + '/poses/'

        self.img_paths = glob.glob('{}{}/image_2/*.png'.format(image_dir, self.folder))
        self.imus = sio.loadmat('{}{}.mat'.format(imu_dir, self.folder))['imu_data_interp']
        self.poses, self.poses_rel = read_pose_from_text('{}{}.txt'.format(pose_dir, self.folder))
        self.img_paths.sort()

        self.img_paths_list, self.poses_list, self.imus_list, self.degradation_labels_list = [], [], [], []
        start = 0
        n_frames = len(self.img_paths)

        while start + self.seq_len < n_frames:
            img_paths_seq = self.img_paths[start:start + self.seq_len]
            poses_seq = self.poses_rel[start:start + self.seq_len - 1]
            imus_seq = self.imus[start * 10:(start + self.seq_len - 1) * 10 + 1]

            # 生成退化标签
            degradation_labels = self.generate_degrade(self.seq_len)
            self.img_paths_list.append(img_paths_seq)
            self.poses_list.append(poses_seq)
            self.imus_list.append(imus_seq)
            self.degradation_labels_list.append(degradation_labels)

            start += self.seq_len - 1

        # 添加剩余数据
        self.img_paths_list.append(self.img_paths[start:])
        self.poses_list.append(self.poses_rel[start:])
        self.imus_list.append(self.imus[start * 10:])
        self.degradation_labels_list.append(self.generate_degrade(len(self.img_paths[start:])))

    def generate_degrade(self, sequence_length):
        # degradation mode
        # 0: normal data 1: occlusion 2: blur 3: image missing 4: imu noise and bias 5: imu missing
        # 6: spatial misalignment 7: temporal misalignment 8: vision degradation 9: all degradation
        # 10: inertial degradation
        degradation_labels = np.zeros(sequence_length)
        for i in range(sequence_length):
            rand_label = np.random.rand()
            if self.data_degradation > 0 and self.data_degradation < 8:
                if rand_label < 0.3:
                    degradation_labels[i] = self.data_degradation
            elif self.data_degradation == 8:  # 视觉退化
                if rand_label < 0.10:
                    degradation_labels[i] = 1
                elif rand_label < 0.20:
                    degradation_labels[i] = 2
                elif rand_label < 0.30:
                    degradation_labels[i] = 3
            elif self.data_degradation == 9:  # 所有退化
                if rand_label < 0.05:
                    degradation_labels[i] = 1
                elif (rand_label > 0.05) and (rand_label < 0.10):
                    degradation_labels[i] = 2
                elif (rand_label > 0.10) and (rand_label < 0.15):
                    degradation_labels[i] = 3
                elif (rand_label > 0.15) and (rand_label < 0.20):
                    degradation_labels[i] = 4
                elif (rand_label > 0.20) and (rand_label < 0.25):
                    degradation_labels[i] = 5
                elif (rand_label > 0.25) and (rand_label < 0.30):
                    degradation_labels[i] = 6
                elif (rand_label > 0.30) and (rand_label < 0.35):
                    degradation_labels[i] = 7
            elif self.data_degradation == 10:   # 惯性退化
                if rand_label < 0.075:
                    degradation_labels[i] = 4
                elif (rand_label > 0.075) and (rand_label < 0.15):
                    degradation_labels[i] = 5
                elif (rand_label > 0.15) and (rand_label < 0.225):
                    degradation_labels[i] = 6
                elif (rand_label > 0.225) and (rand_label < 0.30):
                    degradation_labels[i] = 7
        return degradation_labels

    def degrade_imu_data(self, imu_seq, degradation_label):
        if degradation_label == 4:  # IMU噪声和偏差
            for imu_n, imu in enumerate(imu_seq):
                imu_new = np.copy(imu)
                for k in range(3):
                    imu_new[k] += np.random.rand() * 0.1 + 0.1
                    imu_new[k + 3] += np.random.rand() * 0.001 + 0.001
                # imu_seq[imu_n] = imu_new
            return imu_seq
        elif degradation_label == 5:  # IMU缺失
            imu_seq.fill(0)
            return imu_seq
        if degradation_label == 6:  # Spatial misalignment
            theta = int(np.random.rand(1) * 5) + 5
            rot_theta = np.array([[np.cos(theta), -np.sin(theta), 0],
                                  [np.sin(theta), np.cos(theta), 0],
                                  [0, 0, 1]])
            for imu_n, imu in enumerate(imu_seq):
                imu_new = np.copy(imu)
                imu_new[:3] = imu[:3] @ rot_theta
                imu_new[3:] = imu[3:] @ rot_theta
                imu_seq[imu_n] = imu_new
            return imu_seq
        if degradation_label == 7:  # Temporal misalignment
            for imu_n, imu in enumerate(imu_seq):
                if np.random.rand(1) < 0.5:
                    imu_seq[imu_n] = np.zeros(6).astype(np.float32)
            return imu_seq
        return imu_seq


    def degrade_img(self, img, label):
        img = np.array(img, dtype=np.float32)

        if label == 1:  # 遮挡
            height_start = int(np.random.rand() * 128)
            width_start = int(np.random.rand() * 384)
            img[height_start:height_start + 128, width_start:width_start + 128, :] = 0

        elif label == 2:  # 模糊
            kernel = np.ones((15, 15), np.float32) / 225
            img = cv2.filter2D(img, -1, kernel)
            # 添加盐和胡椒噪声
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(img)
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape]
            out[coords[0], coords[1], coords[2]] = 1

            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape]
            out[coords[0], coords[1], coords[2]] = 0
            img = out

        elif label == 3:
            img = np.zeros((self.opt.img_h, self.opt.img_w, 3)).astype(np.float32)

        return img.astype(np.uint8)


    def __len__(self):
        return len(self.img_paths_list)

    def __getitem__(self, i):
        image_path_sequence = self.img_paths_list[i]
        image_sequence = []
        degradation_labels = self.degradation_labels_list[i]
        imu_sequence = self.imus_list[i]
        gt_sequence = self.poses_list[i][:, :6]

        for j, (img_path, label) in enumerate(zip(image_path_sequence, degradation_labels)):
            img = Image.open(img_path).convert('RGB')
            img = TF.resize(img, size=(self.opt.img_h, self.opt.img_w))
            img = self.degrade_img(img, int(label))

            if j < self.seq_len-1:
                imu_sequence[j*10:j*10+11] = self.degrade_imu_data(imu_sequence[j*10:j*10+11], int(label))

            img_as_tensor = TF.to_tensor(img) - 0.5
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)

        image_sequence = torch.cat(image_sequence, 0)

        return image_sequence, torch.FloatTensor(imu_sequence), gt_sequence


class KITTI_tester():
    def __init__(self, args):
        super(KITTI_tester, self).__init__()

        # generate data loader for each path
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq, args.test_degradation_mode))

        self.args = args

    def test_one_path(self, net, random, df, num_gpu=1):
        hc, ncp_hc = None, None
        pose_list, decision_list, selection_list = [], [], []
        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
            # Adjust the batch size and device based on available GPUs
            image_seq = image_seq.unsqueeze(0).to(self.args.device)
            imu_seq = imu_seq.unsqueeze(0).to(self.args.device)

            if num_gpu > 1:
                image_seq = image_seq.repeat(num_gpu, 1, 1, 1, 1)
                imu_seq = imu_seq.repeat(num_gpu, 1, 1)

            with torch.no_grad():
                pose, hc, ncp_hc, decision, selection = net(image_seq, imu_seq, is_first=(i == 0), hc=hc, ncp_hc=ncp_hc,
                                                            random=random)
            pose_list.append(pose[0, :, :].detach().cpu().numpy())
            decision_list.append(decision[0, :, :].detach().cpu().numpy()[:, 0])
            selection_list.append(selection[0, :, :].detach().cpu().numpy()[:, 0])

        pose_est = np.vstack(pose_list)
        dec_est = np.hstack(decision_list)
        sel_est = np.hstack(selection_list)
        return pose_est, dec_est, sel_est

    def eval(self, net, random=False, num_gpu=1):
        self.errors = []
        self.est = []
        for i, seq in enumerate(self.args.val_seq):
            print(f'testing sequence {seq}')
            pose_est, dec_est, sel_est = self.test_one_path(net, random, self.dataloader[i], num_gpu=num_gpu)
            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed, d_usage, s_usage = kitti_eval(
                pose_est, dec_est, sel_est, self.dataloader[
                    i].poses_rel)

            self.est.append({'pose_est_global': pose_est_global, 'pose_gt_global': pose_gt_global, 'speed': speed,
                             'decision': dec_est, 'selection': sel_est})
            self.errors.append({'t_rel': t_rel, 'r_rel': r_rel, 't_rmse': t_rmse, 'r_rmse': r_rmse, 'd_usage': d_usage,
                                's_usage': s_usage})

        return self.errors

    def generate_plots(self, save_dir, window_size):
        for i, seq in enumerate(self.args.val_seq):
            degradation_labels = self.dataloader[i].degradation_labels_list
            plotPath_2D(seq,
                        self.est[i]['pose_gt_global'],
                        self.est[i]['pose_est_global'],
                        save_dir,
                        self.est[i]['speed'],
                        self.est[i]['decision'],
                        self.est[i]['selection'],
                        degradation_labels,
                        window_size)

    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir / '{}_pred.txt'.format(seq)
            saveSequence(self.est[i]['pose_est_global'], path)
            print('Seq {} saved'.format(seq))


def kitti_eval(pose_est, dec_est, sel_est, pose_gt):
    # First decision is always true
    dec_est = np.insert(dec_est, 0, 1)
    sel_est = np.insert(sel_est, 0, 1)

    # Calculate the translational and rotational RMSE
    t_rmse, r_rmse = rmse_err_cal(pose_est, pose_gt)

    # Transfer to 3x4 pose matrix
    pose_est_mat = path_accu(pose_est)
    pose_gt_mat = path_accu(pose_gt)

    # Using KITTI metric
    err_list, t_rel, r_rel, speed = kitti_err_cal(pose_est_mat, pose_gt_mat)

    t_rel = t_rel * 100
    r_rel = r_rel / np.pi * 180 * 100
    r_rmse = r_rmse / np.pi * 180
    d_usage = np.mean(dec_est) * 100
    s_usage = np.mean(sel_est) * 100
    return pose_est_mat, pose_gt_mat, t_rel, r_rel, t_rmse, r_rmse, speed, d_usage, s_usage


def kitti_err_cal(pose_est_mat, pose_gt_mat):
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    num_lengths = len(lengths)

    err = []
    dist, speed = trajectoryDistances(pose_gt_mat)
    step_size = 10  # 10Hz

    for first_frame in range(0, len(pose_gt_mat), step_size):

        for i in range(num_lengths):
            len_ = lengths[i]
            last_frame = lastFrameFromSegmentLength(dist, first_frame, len_)
            # Continue if sequence not long enough
            if last_frame == -1 or last_frame >= len(pose_est_mat) or first_frame >= len(pose_est_mat):
                continue

            pose_delta_gt = np.dot(np.linalg.inv(pose_gt_mat[first_frame]), pose_gt_mat[last_frame])
            pose_delta_result = np.dot(np.linalg.inv(pose_est_mat[first_frame]), pose_est_mat[last_frame])

            r_err = rotationError(pose_delta_result, pose_delta_gt)
            t_err = translationError(pose_delta_result, pose_delta_gt)

            err.append([first_frame, r_err / len_, t_err / len_, len_])

    t_rel, r_rel = computeOverallErr(err)
    return err, t_rel, r_rel, np.asarray(speed)


def plotPath_2D(seq, poses_gt_mat, poses_est_mat, plot_path_dir, speed, decision, selection, degradation_labels, window_size):
    # Apply smoothing to the decision
    decision = np.insert(decision, 0, 1)
    selection = np.insert(selection, 0, 1)

    d_or_s = np.zeros_like(decision, dtype=float)
    for i in range(decision.shape[0]):
        if decision[i] == 0:
            if selection[i] == 1:
                d_or_s[i] = 0.0
            else:
                d_or_s[i] = 1.0
        else:
            d_or_s[i] = 0.5
    dos = d_or_s
    d_or_s = moving_average(d_or_s, window_size)


    fontsize_ = 10
    plot_keys = ["Ground Truth", "Ours"]
    start_point = [0, 0]
    style_pred = 'b-'
    style_gt = 'r-'
    style_O = 'ko'

    # get the value
    x_gt = np.asarray([pose[0, 3] for pose in poses_gt_mat])
    y_gt = np.asarray([pose[1, 3] for pose in poses_gt_mat])
    z_gt = np.asarray([pose[2, 3] for pose in poses_gt_mat])

    x_pred = np.asarray([pose[0, 3] for pose in poses_est_mat])
    y_pred = np.asarray([pose[1, 3] for pose in poses_est_mat])
    z_pred = np.asarray([pose[2, 3] for pose in poses_est_mat])

    save_trajectory(x_pred, z_pred, plot_path_dir, 'pred', seq)
    save_trajectory(x_gt, z_gt, plot_path_dir, 'gt', seq)

    ##############################    Plot 2d trajectory estimation map   ###############################################
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.gca()
    plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
    plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
    plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
    plt.legend(loc="upper right", prop={'size': fontsize_})
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    # set the range of x and y
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean))
                       for lim in lims])
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    plt.title('2D path')
    png_title = "{}_path_2d".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    ##############################    Plot degradation heatmap   ##################################################
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.gca()
    # 规范处理
    dlabels = np.hstack(degradation_labels)
    dlabels = standardize_array(dlabels)    # Now dlabels.shape = (n, )

    dlabels_float = np.zeros_like(dlabels, dtype=float)
    # 0: Normal, 1-3: Visual Noise, 4-7: IMU Noise
    dlabels_float[(dlabels == 0)] = 0.5
    dlabels_float[(dlabels >= 1) & (dlabels <= 3)] = 1.0
    dlabels_float[(dlabels >= 4) & (dlabels <= 7)] = 0.0
    unique_values, counts = np.unique(dlabels_float, return_counts=True)
    # 计算百分比
    percentages = counts / len(dlabels_float) * 100
    # 输出结果
    msg = f"In seq {seq} |"
    for value, percentage in zip(unique_values, percentages):
        if value == 0.5:
            value = "normal"
        elif value == 1.0:
            value = "visual degradation"
        else:
            value = "inertial degradation"
        
        msg += f" {value} : {percentage:.2f}% |"
    # print(msg)
    msg += "\n          "

    dlabels_float = moving_average(dlabels_float, window_size)
    
    cout = dlabels_float * 100
    # Create scatter plot with colors based on degradation labels
    cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)

    # Set axis limits based on path
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    # 连续
    max_usage = max(cout)
    min_usage = min(cout)
    ticks = np.floor(np.linspace(min_usage, max_usage, num=3))
    cbar = fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels([str(i) + '%' for i in ticks])

    plt.title('Degradation Heatmap')
    png_title = "{}_degradation_heatmap".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    ##############################    Plot check traj map   ###############################################
    # fig = plt.figure(figsize=(8, 6), dpi=100)
    # ax = plt.gca()

    # 标准化处理
    dlabels = np.hstack(degradation_labels)
    dlabels = standardize_array(dlabels)    # Now dlabels.shape = (n, )

    dlabels_float = np.zeros_like(dlabels, dtype=float)
    # 0: Normal, 1-3: Visual Noise, 4-7: IMU Noise
    dlabels_float[(dlabels == 0)] = 1   # 正常
    dlabels_float[(dlabels >= 1) & (dlabels <= 3)] = 0.5  # 该使用惯性
    dlabels_float[(dlabels >= 4) & (dlabels <= 7)] = 0  # 该使用视觉

    is_fusion_label = np.insert(decision, 0, 0)    
    is_visual_label = np.insert(selection, 0, 0)
    data_len = dlabels_float.shape[0]
    check = np.zeros_like(dlabels_float, dtype=float)
    
    # 统计错误法
    for i in range(data_len):
        if i == 0:
            check[i] = 0
        if dlabels_float[i] == 0.5 and is_visual_label[i] == 1:
            check[i] = 1
        elif dlabels_float[i] == 0 and is_visual_label[i] == 0:
            check[i] = 1
        else:
            check[i] = 0
    unique_values, counts = np.unique(check, return_counts=True)
    # 计算百分比
    percentages = counts / len(check) * 100
    # 输出结果
    for value, percentage in zip(unique_values, percentages):
        value = "*Right*" if value == 0 else "*Wrong*"
        msg+= f"{value}={percentage:.2f}%   "
    print(msg)
    # # # 平滑检验
    # # check = moving_average(check, window_size)
    # # cout = check*100

    # # # 平滑退化
    # # dlabels_float = smooth_array3(dlabels_float)
    # # cout = dlabels_float

    # # 离散检验
    # color_map = {
    #     0: '#32CD32',   # 正确
    #     1: '#ff2a00',   # 错误
    # }
    # color_values = [color_map.get(c, 'white') for c in check]

    # # 离散
    # cax = plt.scatter(x_pred, z_pred, marker='o', c=color_values)
    # # # 连续
    # # cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
    # plt.xlabel('x (m)', fontsize=fontsize_)
    # plt.ylabel('z (m)', fontsize=fontsize_)

    # # Set axis limits based on path
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # xmean = np.mean(xlim)
    # ymean = np.mean(ylim)
    # ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    # ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    # # 离散
    # cbar = fig.colorbar(cax)
    # cbar.ax.set_yticklabels([str(k) for k in color_map.keys()])

    # # # 连续
    # # max_usage = max(cout)
    # # min_usage = min(cout)
    # # ticks = np.floor(np.linspace(min_usage, max_usage, num=3))
    # # cbar = fig.colorbar(cax, ticks=ticks)
    # # # cbar.ax.set_yticklabels([str(i) + '%' for i in ticks])

    # plt.title('Check traj')
    # png_title = "{}_check".format(seq)
    # plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    # plt.close()

    ##################################    Plot scatter map   ##############################################
    # d_or_s: 0=visual  0.5=fusion  1=inertial
    # remap
    # label_map_A = {0: "visual degradation", 0.5: "normal data", 1: "inertial degradation"}
    # label_map_B = {0: "visual modal", 0.5: "fusion", 1: "inertial modal"}
    A = np.zeros_like(dlabels, dtype=float)
    A[(dlabels == 0)] = 0.5
    A[(dlabels >= 1) & (dlabels <= 3)] = 1.0
    A[(dlabels >= 4) & (dlabels <= 7)] = 0.0
    B = dos
    B = np.insert(B, 0, 0.5)
    # print(A.shape, B.shape)
    # y1, y2 = find_max_correct_window_fixed_length(A, B, 256)
    # print(y1,y2)
    # exit()
    # 150: 488 638
    # 256: 488 744
    A = A[488:638]
    B = B[488:638]
    n = 150
    time = np.arange(n)
    # 颜色映射
    # correct_1 = (A == 0.5) & (B == 0.5)  # A=0.5, B=0.5 (正确情况)
    # correct_2 = (A == 1) & (B == 1)  # A=1, B=1 (正确情况)
    # correct_3 = (A == 0) & (B == 0)  # A=0, B=0 (正确情况)
    # is_ok_1 = (A == 0) & (B == 0.5)
    # is_ok_2 = (A == 1) & (B == 0.5)
    # is_ok_3 = (A == 0.5) & (B == 0)
    # is_ok_4 = (A == 0.5) & (B == 1)
    # wrong_1 = (A == 0) & (B == 1)  # 错误情况
    # wrong_2 = (A == 1) & (B == 0)  # 错误情况

    correct_1 = (A == 0.5) & (B == 0.5)
    correct_2 = (A == 1) & (B == 0)  
    correct_3 = (A == 0) & (B == 1)  
    is_ok_1 = (A == 0) & (B == 0.5)
    is_ok_2 = (A == 1) & (B == 0.5)
    is_ok_3 = (A == 0.5) & (B == 0)
    is_ok_4 = (A == 0.5) & (B == 1)
    wrong_1 = (A == 1) & (B == 1)  # 错误情况
    wrong_2 = (A == 0) & (B == 0)  # 错误情况

    colors = np.full(n, 'gray', dtype='<U6')  # 默认灰色
    colors[correct_1 | correct_2 | correct_3] = 'green'   # 正确情况
    colors[is_ok_1 | is_ok_2 | is_ok_3 | is_ok_4] = 'orange'  # 可接受情况
    colors[wrong_1 | wrong_2] = 'red'  # 错误情况


    # 创建图像
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # 重新映射 A 和 B 的值，使 A 轴在上方，B 轴在下方
    # A_mapped = np.where(A == 0.5, 0.8, np.where(A == 1, 1.4, 0.2))
    # B_mapped = np.where(B == 0.5, 0.7, np.where(B == 1, 1.3, 0.1))

    A_mapped = np.where(A == 0.5, 0.6, np.where(A == 1, 1.0, 0.2))
    B_mapped = np.where(B == 0.5, 0.5, np.where(B == 1, 0.9, 0.1))

    # 设置左轴（A）的 Y 轴标签
    ax1.set_yticks([0.2, 0.6, 1.0])  
    ax1.set_yticklabels(["Visual Degradation", "Normal Data", "Inertial Degradation"])
    ax1.set_ylabel("A values (Degradation levels)", fontsize=12, color='blue')

    # 右轴（B）与左轴错开
    ax2 = ax1.twinx()
    ax2.set_yticks([0.1, 0.5, 0.9])  
    ax2.set_yticklabels(["Visual Modal", "Fusion", "Inertial Modal"])
    ax2.set_ylabel("B values (Selection modes)", fontsize=12, color='purple')

    # 手动设置Y轴范围，解决顶部对齐问题
    # ax1.set_ylim(0, 1.5)  # 左侧轴范围扩展，使1.4不在顶部
    # ax2.set_ylim(0, 1.5)  # 右侧轴保持原范围

    ax1.set_ylim(0, 1.1)  # 左侧轴范围扩展，使1.4不在顶部
    ax2.set_ylim(0, 1.1)  # 右侧轴保持原范围

    # 画背景区域以突出正确选择
    ax1.axhspan(0.1, 0.2, color='lightcoral', alpha=0.3)  # Visual Degradation 正确区
    ax1.axhspan(0.5, 0.6, color='lightgreen', alpha=0.3)  # Normal Data 正确区
    ax1.axhspan(0.9, 1.0, color='lightcoral', alpha=0.3)  # Inertial Degradation 正确区

    # 画散点图
    ax1.scatter(time, A_mapped, c=colors, marker='o', alpha=0.7, label="A values", s=50, edgecolors='black')
    ax2.scatter(time, B_mapped, c=colors, marker='s', alpha=0.7, label="B values", s=50, edgecolors='black')

    # 纵向连接正确的 A 和 B 数据点
    for i in range(n):
        if colors[i] == 'green':  # 仅在正确选择的情况下绘制连接线
            ax1.plot([time[i], time[i]], [B_mapped[i], A_mapped[i]], color='green', linestyle='-', linewidth=2, alpha=0.8)
        elif colors[i] == 'red': 
            ax1.plot([time[i], time[i]], [B_mapped[i], A_mapped[i]], color='red', linestyle='-', linewidth=1.5, alpha=0.8)
        elif colors[i] == 'orange':  
            ax1.plot([time[i], time[i]], [B_mapped[i], A_mapped[i]], color='orange', linestyle='-', linewidth=1.5, alpha=0.8)

    # 增加水平虚线以对齐 A 和 B 的水平关系
    for y in [0.2, 0.6, 1.0]:  # A 轴参考线
        ax1.hlines(y, xmin=0, xmax=n, color='gray', linestyle='dashed', alpha=0.5)
    for y in [0.1, 0.5, 0.9]:  # B 轴参考线
        ax2.hlines(y, xmin=0, xmax=n, color='gray', linestyle='dashed', alpha=0.5)

    ax1.set_xlabel("Time step", fontsize=12)
    ax1.set_title("Time Series Scatter Plot with Vertical Links for Correct Selections", fontsize=14, fontweight='bold')
    legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Degradation mode',
               markerfacecolor='lightgray', markersize=10, markeredgecolor='black'),
    plt.Line2D([0], [0], marker='s', color='w', label='Model decision',
               markerfacecolor='lightgray', markersize=10, markeredgecolor='black'),
    plt.Line2D([0], [0], color='green', lw=2, label='Correct'),
    plt.Line2D([0], [0], color='orange', lw=2, label='Acceptable'),
    plt.Line2D([0], [0], color='red', lw=2, label='Incorrect')
    ]

    legend = ax1.legend(handles=legend_elements, loc='lower left',bbox_to_anchor=(1.05, 1), fontsize=12, frameon=True, framealpha=1)
    legend.set_zorder(10)  # 提升图例层级

    fig.tight_layout()
    png_title = "{}_analysis".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", dpi=300)
    plt.close()


    ##########################    Plot decision & selection heatmap   ######################################
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.gca()
    cout = np.insert(d_or_s, 0, 1) * 100
    cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
    max_usage = max(cout)
    min_usage = min(cout)
    ticks = np.floor(np.linspace(min_usage, max_usage, num=3))
    cbar = fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels([str(i) + '%' for i in ticks])

    plt.title('d_or_s heatmap with window size {}'.format(window_size))
    png_title = "{}_d_or_s_smoothed".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    ################################    Plot the speed map   #################################################
    # Plot the speed map
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.gca()
    cout = speed
    cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
    max_speed = max(cout)
    min_speed = min(cout)
    ticks = np.floor(np.linspace(min_speed, max_speed, num=5))
    cbar = fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels([str(i) + 'm/s' for i in ticks])

    plt.title('speed heatmap')
    png_title = "{}_speed".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

