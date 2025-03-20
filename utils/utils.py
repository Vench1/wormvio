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
from prettytable import PrettyTable
from collections import deque
plt.switch_backend('agg')

_EPS = np.finfo(float).eps * 4.0

def table_calculate_size(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            table.add_row([name, parameter.size()])
            total_params += parameter.numel()
    print(table)
    return total_params

def isRotationMatrix(R):
    '''
    check whether a matrix is a qualified rotation metrix
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def euler_from_matrix(matrix):
    '''
    Extract the eular angle from a rotation matrix
    '''
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
    ay = math.atan2(-M[2, 0], cy)
    if ay < -math.pi / 2 + _EPS and ay > -math.pi / 2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2(-M[1, 2], -M[0, 2])
    elif ay < math.pi / 2 + _EPS and ay > math.pi / 2 - _EPS:
        ax = 0
        az = math.atan2(M[1, 2], M[0, 2])
    else:
        ax = math.atan2(M[2, 1], M[2, 2])
        az = math.atan2(M[1, 0], M[0, 0])
    return np.array([ax, ay, az])

def get_relative_pose(Rt1, Rt2):
    '''
    Calculate the relative 4x4 pose matrix between two pose matrices
    '''
    Rt1_inv = np.linalg.inv(Rt1)
    Rt_rel = Rt1_inv @ Rt2
    return Rt_rel

def get_relative_pose_6DoF(Rt1, Rt2):
    '''
    Calculate the relative rotation and translation from two consecutive pose matrices 
    '''
    
    # Calculate the relative transformation Rt_rel
    Rt_rel = get_relative_pose(Rt1, Rt2)

    R_rel = Rt_rel[:3, :3]
    t_rel = Rt_rel[:3, 3]

    # Extract the Eular angle from the relative rotation matrix
    x, y, z = euler_from_matrix(R_rel)
    theta = [x, y, z]

    pose_rel = np.concatenate((theta, t_rel))
    return pose_rel

def rotationError(Rt1, Rt2):
    '''
    Calculate the rotation difference between two pose matrices
    '''
    pose_error = get_relative_pose(Rt1, Rt2)
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))

def translationError(Rt1, Rt2):
    '''
    Calculate the translational difference between two pose matrices
    '''
    pose_error = get_relative_pose(Rt1, Rt2)
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    return np.sqrt(dx**2 + dy**2 + dz**2)

def eulerAnglesToRotationMatrix(theta):
    '''
    Calculate the rotation matrix from eular angles (roll, yaw, pitch)
    '''
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def normalize_angle_delta(angle):
    '''
    Normalization angles to constrain that it is between -pi and pi
    '''
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

def pose_6DoF_to_matrix(pose):
    '''
    Calculate the 3x4 transformation matrix from Eular angles and translation vector
    '''
    R = eulerAnglesToRotationMatrix(pose[:3])
    t = pose[3:].reshape(3, 1)
    R = np.concatenate((R, t), 1)
    R = np.concatenate((R, np.array([[0, 0, 0, 1]])), 0)
    return R

def pose_accu(Rt_pre, R_rel):
    '''
    Calculate the accumulated pose from the latest pose and the relative rotation and translation
    '''
    Rt_rel = pose_6DoF_to_matrix(R_rel)
    return Rt_pre @ Rt_rel

def path_accu(pose):
    '''
    Generate the global pose matrices from a series of relative poses
    '''
    answer = [np.eye(4)]
    for index in range(pose.shape[0]):
        pose_ = pose_accu(answer[-1], pose[index, :])
        answer.append(pose_)
    return answer

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def rmse_err_cal(pose_est, pose_gt):
    '''
    Calculate the rmse of relative translation and rotation
    '''
    t_rmse = np.sqrt(np.mean(np.sum((pose_est[:, 3:] - pose_gt[:, 3:])**2, -1)))
    r_rmse = np.sqrt(np.mean(np.sum((pose_est[:, :3] - pose_gt[:, :3])**2, -1)))
    return t_rmse, r_rmse

def trajectoryDistances(poses):
    '''
    Calculate the distance and speed for each frame
    '''
    dist = [0]
    speed = [0]
    for i in range(len(poses) - 1):
        cur_frame_idx = i
        next_frame_idx = cur_frame_idx + 1
        P1 = poses[cur_frame_idx]
        P2 = poses[next_frame_idx]
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dz = P1[2, 3] - P2[2, 3]
        dist.append(dist[i] + np.sqrt(dx**2 + dy**2 + dz**2))
        speed.append(np.sqrt(dx**2 + dy**2 + dz**2) * 10)
    return dist, speed

def lastFrameFromSegmentLength(dist, first_frame, len_):
    for i in range(first_frame, len(dist), 1):
        if dist[i] > (dist[first_frame] + len_):
            return i
    return -1

def computeOverallErr(seq_err):
    t_err = 0
    r_err = 0
    seq_len = len(seq_err)

    for item in seq_err:
        r_err += item[1]
        t_err += item[2]
    ave_t_err = t_err / seq_len
    ave_r_err = r_err / seq_len
    return ave_t_err, ave_r_err

def read_pose(line):
    '''
    Reading 4x4 pose matrix from .txt files
    input: a line of 12 parameters
    output: 4x4 numpy matrix
    '''
    values= np.reshape(np.array([float(value) for value in line.split(' ')]), (3, 4))
    Rt = np.concatenate((values, np.array([[0, 0, 0, 1]])), 0)
    return Rt
    
def read_pose_from_text(path):
    with open(path) as f:
        lines = [line.split('\n')[0] for line in f.readlines()]
        poses_rel, poses_abs = [], []
        values_p = read_pose(lines[0])
        poses_abs.append(values_p)            
        for i in range(1, len(lines)):
            values = read_pose(lines[i])
            poses_rel.append(get_relative_pose_6DoF(values_p, values)) 
            values_p = values.copy()
            poses_abs.append(values) 
        poses_abs = np.array(poses_abs)
        poses_rel = np.array(poses_rel)
        
    return poses_abs, poses_rel

def saveSequence(poses, file_name):
    with open(file_name, 'w') as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            f.write(' '.join([str(r) for r in pose]))
            f.write('\n')

def standardize_array(arr):
    n = arr.size
    m = (n + 10) // 11  # 计算总共有多少组，相当于ceil(n/11)
    processed_groups = []
    for i in range(m):
        start = i * 11
        end = start + 11
        if end > n:
            end = n
        group = arr[start:end]
        if i < m - 1:
            # 前面的组取前10个元素
            processed_groups.append(group[:-1])
        else:
            # 最后一组保留所有元素
            processed_groups.append(group)
    return np.concatenate(processed_groups)

def smooth_array3(arr, par=0.4):
    n = len(arr)
    output = np.zeros(n)

    # 识别所有连续的非零块
    blocks = []
    i = 0
    while i < n:
        if arr[i] != 0:
            start = i
            while i < n and arr[i] != 0:
                i += 1
            end = i - 1
            avg = np.mean(arr[start:end + 1])
            blocks.append((start, end, avg))
        else:
            i += 1

    # 对每个块生成高斯波并叠加
    for start, end, avg in blocks:
        length = end - start + 1
        center = (start + end) / 2.0
        sigma = max(0.8, length * par)  # 调整标准差系数以控制平滑程度
        x = np.arange(n)
        kernel = avg * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        output += kernel

    # 裁剪到0-1之间并返回
    output = np.clip(output, 0, 1)
    return output

def save_trajectory(x, z, save_dir, traj_type, index=None):
    """保存轨迹数据到本地"""
    os.makedirs(save_dir, exist_ok=True)
    
    if traj_type == 'gt':
        filename = os.path.join(save_dir, f'gt_traj_{index}.txt')
    elif traj_type == 'pred':
        filename = os.path.join(save_dir, f'pred_traj_{index}.txt')
    else:
        raise ValueError("traj_type must be 'gt' or 'pred'")
    
    # 合并x,z坐标并保存
    trajectory = np.column_stack((x, z))
    np.savetxt(filename, trajectory, fmt='%.6f')

def plot_trajectories(gt_path, pred_paths, names, save_path, 
                     gt_style='k-', pred_styles=None, 
                     start_style='ko', title='2D Trajectory Comparison'):
    """可视化并保存轨迹对比图"""
    # 加载数据
    gt_data = np.loadtxt(gt_path)
    x_gt, z_gt = gt_data[:, 0], gt_data[:, 1]
    
    pred_trajs = []
    for path in pred_paths:
        data = np.loadtxt(path)
        pred_trajs.append((data[:, 0], data[:, 1]))
    
    # 创建画布
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.gca()
    
    # 绘制轨迹
    plt.plot(x_gt, z_gt, gt_style, label='Ground Truth')
    
    # 绘制所有预测轨迹
    colors = plt.cm.tab10.colors  # 使用tab10色卡
    for i, (x_pred, z_pred) in enumerate(pred_trajs):
        style = pred_styles[i] if pred_styles and i < len(pred_styles) else f'{colors[i % 10]}-'
        plt.plot(x_pred, z_pred, style, label=f'{names[i]}')
    
    # 绘制起点
    start_point = (x_gt[0], z_gt[0])
    plt.plot(start_point[0], start_point[1], start_style, label='Start Point')
    
    # 设置图形属性
    plt.legend(loc="upper right", prop={'size': 10})
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    plt.title(title)
    
    # 自动调整坐标轴范围
    all_coords = np.vstack([gt_data] + [np.column_stack(p) for p in pred_trajs])
    x_mean, z_mean = np.mean(all_coords, axis=0)
    max_offset = np.max(np.abs(all_coords - [x_mean, z_mean]))
    
    ax.set_xlim([x_mean - max_offset, x_mean + max_offset])
    ax.set_ylim([z_mean - max_offset, z_mean + max_offset])
    ax.set_aspect('equal', adjustable='datalim')
    
    # 保存结果
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def find_max_correct_window_fixed_length(A, B, x):
    n = A.shape[0]
    if x <= 0 or x > n:
        raise ValueError("x must be a positive integer less than or equal to n")
    
    # 生成正确标志数组，1表示A[i] == B[i]，否则为0
    C = (A == B).astype(int)
    
    max_sum = -1
    y1, y2 = 0, x - 1  # 初始窗口为[0, x-1]
    
    # 计算初始窗口的正确数总和
    current_sum = np.sum(C[:x])
    max_sum = current_sum
    
    # 滑动窗口遍历所有可能的起始位置
    for i in range(1, n - x + 1):
        # 减去滑出窗口的元素，加上新进入窗口的元素
        current_sum = current_sum - C[i-1] + C[i + x - 1]
        # 更新最大值和对应的窗口
        if current_sum > max_sum:
            max_sum = current_sum
            y1, y2 = i, i + x - 1
        # 如果相等，保留最左边的窗口（即不更新）
    
    return y1, y2