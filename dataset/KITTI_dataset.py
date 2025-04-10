import sys

sys.path.append('..')
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from path import Path
from utils.utils import rotationError, read_pose_from_text
from utils import custom_transform
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
import random
import cv2
import torchvision.transforms.functional as TF

IMU_FREQ = 10


class KITTI(Dataset):
    def __init__(self, root,
                 sequence_length=11,
                 train_seqs=['00', '01', '02', '04', '06', '08', '09'],
                 transform=None,
                 data_degradation=0):

        self.root = Path(root)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train_seqs = train_seqs
        self.data_degradation = data_degradation
        self.make_dataset()

    def make_dataset(self):
        sequence_set = []
        for folder in self.train_seqs:
            poses, poses_rel = read_pose_from_text(self.root / 'poses/{}.txt'.format(folder))
            imus = sio.loadmat(self.root / 'imus/{}.mat'.format(folder))['imu_data_interp']
            fpaths = sorted((self.root / 'sequences/{}/image_2'.format(folder)).files("*.png"))
            for i in range(len(fpaths) - self.sequence_length):
                img_samples = fpaths[i:i + self.sequence_length]
                imu_samples = imus[i * IMU_FREQ:(i + self.sequence_length - 1) * IMU_FREQ + 1]
                pose_samples = poses[i:i + self.sequence_length]
                pose_rel_samples = poses_rel[i:i + self.sequence_length - 1]
                segment_rot = rotationError(pose_samples[0], pose_samples[-1])

                sample = {
                    'imgs': img_samples,
                    'imus': imu_samples,
                    'gts': pose_rel_samples,
                    'rot': segment_rot,
                    'data_degradation': np.zeros(self.sequence_length).tolist()  # Initialize degradation array
                }

                self.generate_degrade(sample)
                self.degrade_imu_data(sample)

                sequence_set.append(sample)
        self.samples = sequence_set

        # Generate weights based on the rotation of the training segments
        rot_list = np.array([np.cbrt(item['rot'] * 180 / np.pi) for item in self.samples])
        rot_range = np.linspace(np.min(rot_list), np.max(rot_list), num=10)
        indexes = np.digitize(rot_list, rot_range, right=False)
        num_samples_of_bins = dict(Counter(indexes))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(1, len(rot_range) + 1)]

        lds_kernel_window = self.get_lds_kernel_window(kernel='gaussian', ks=7, sigma=5)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

        self.weights = [np.float32(1 / eff_label_dist[bin_idx - 1]) for bin_idx in indexes]

    def generate_degrade(self, sample):
        # degradation mode
        # 0: normal data 1: occlusion 2: blur 3: image missing 4: imu noise and bias 5: imu missing
        # 6: spatial misalignment 7: temporal misalignment 8: vision degradation 9: all degradation
        # 10：inertial degradation
        for i in range(self.sequence_length):
            rand_label = np.random.rand(1)
            if self.data_degradation == 0:
                return

            if (self.data_degradation > 0) and (self.data_degradation < 8):
                if rand_label < 0.3:
                    sample['data_degradation'][i] = self.data_degradation

            if self.data_degradation == 8:
                if rand_label < 0.10:
                    sample['data_degradation'][i] = 1
                elif rand_label < 0.20:
                    sample['data_degradation'][i] = 2
                elif rand_label < 0.30:
                    sample['data_degradation'][i] = 3

            if self.data_degradation == 9:
                if rand_label < 0.05:
                    sample['data_degradation'][i] = 1
                elif (rand_label > 0.05) and (rand_label < 0.10):
                    sample['data_degradation'][i] = 2
                elif (rand_label > 0.10) and (rand_label < 0.15):
                    sample['data_degradation'][i] = 3
                elif (rand_label > 0.15) and (rand_label < 0.20):
                    sample['data_degradation'][i] = 4
                elif (rand_label > 0.20) and (rand_label < 0.25):
                    sample['data_degradation'][i] = 5
                elif (rand_label > 0.25) and (rand_label < 0.30):
                    sample['data_degradation'][i] = 6
                elif (rand_label > 0.30) and (rand_label < 0.35):
                    sample['data_degradation'][i] = 7

            if self.data_degradation == 10:
                if rand_label < 0.075:
                    sample['data_degradation'][i] = 4
                elif (rand_label > 0.075) and (rand_label < 0.15):
                    sample['data_degradation'][i] = 5
                elif (rand_label > 0.15) and (rand_label < 0.225):
                    sample['data_degradation'][i] = 6
                elif (rand_label > 0.225) and (rand_label < 0.30):
                    sample['data_degradation'][i] = 7

    def degrade_imu_data(self, sample):
        # imus : len=101
        for i in range(self.sequence_length-1):
            if sample['data_degradation'][i] == 4:  # IMU noise and bias
                imu_seq = sample['imus'][i*10:i*10+11]
                for imu_n, imu in enumerate(imu_seq):
                    imu_new = np.copy(imu)
                    for k in range(3):
                        imu_new[k] += np.random.rand(1) * 0.1 + 0.1
                        imu_new[k + 3] += np.random.rand(1) * 0.001 + 0.001
                    imu_seq[imu_n] = imu_new
                sample['imus'][i*10:i*10+11] = imu_seq

            if sample['data_degradation'][i] == 5:  # IMU missing
                sample['imus'][i*10:i*10+11] = np.zeros((11, 6)).astype(np.float32)

            if sample['data_degradation'][i] == 6:  # Spatial misalignment
                theta = int(np.random.rand(1) * 5) + 5
                rot_theta = np.array([[np.cos(theta), -np.sin(theta), 0],
                                      [np.sin(theta), np.cos(theta), 0],
                                      [0, 0, 1]])
                imu_seq = sample['imus'][i * 10:i * 10 + 11]
                for imu_n, imu in enumerate(imu_seq):
                    imu_new = np.copy(imu)
                    imu_new[:3] = imu[:3] @ rot_theta
                    imu_new[3:] = imu[3:] @ rot_theta
                    imu_seq[imu_n] = imu_new
                sample['imus'][i * 10:i * 10 + 11] = imu_seq

            if sample['data_degradation'][i] == 7:  # Temporal misalignment
                imu_seq = sample['imus'][i * 10:i * 10 + 11]
                for imu_n, imu in enumerate(imu_seq):
                    if np.random.rand(1) < 0.5:
                        imu_seq[imu_n] = np.zeros(6).astype(np.float32)
                sample['imus'][i * 10:i * 10 + 11] = imu_seq

    def __getitem__(self, index):
        sample = self.samples[index]

        imgs = []
        for img_path, label in zip(sample['imgs'], sample['data_degradation']):
            img = self.load_as_float(img_path, label)
            imgs.append(img)

        if self.transform is not None:
            imgs, imus, gts = self.transform(imgs, np.copy(sample['imus']), np.copy(sample['gts']))
        else:
            imus = np.copy(sample['imus'])
            gts = np.copy(sample['gts']).astype(np.float32)

        rot = sample['rot'].astype(np.float32)
        weight = self.weights[index]

        return imgs, imus, gts, rot, weight

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Training sequences: '
        for seq in self.train_seqs:
            fmt_str += '{} '.format(seq)
        fmt_str += '\n'
        fmt_str += '    Number of segments: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def get_lds_kernel_window(self, kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
                gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks)
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
                map(laplace, np.arange(-half_ks, half_ks + 1)))

        return kernel_window

    def load_as_float(self, path, label):
        img = Image.open(path).convert('RGB')
        img = TF.resize(img, size=(256, 512))
        img = np.array(img, dtype=np.float32)

        if label == 1:
            height_start = int(np.random.rand(1) * 128)

            width_start = int(np.random.rand(1) * 384)

            for ind_h in range(height_start, height_start + 128):
                for ind_w in range(width_start, width_start + 128):
                    for ind_c in range(0, 3):
                        img[ind_h, ind_w, ind_c] = 0

        if label == 3:
            img = np.zeros((256, 512, 3)).astype(np.float32)

        if label == 2:
            kernel = np.ones((15, 15), np.float32) / 225
            img = cv2.filter2D(img, -1, kernel)

            row, col, ch = img.shape
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

        return img.astype(np.uint8)
