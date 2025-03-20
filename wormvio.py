import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.timer import Timer

from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )


class Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))
        self.proj = nn.Linear(256 * 1 * (opt.imu_len + 1), opt.i_f_len)

    def forward(self, x):
        # x: (B, seq_len, imu_len + 1, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))  # x: (B * seq_len, imu_len + 1, 6)
        x = self.encoder_conv(x.permute(0, 2, 1))  # x: (B * seq_len, 256, imu_len + 1)
        out = self.proj(x.view(x.shape[0], -1))  # out: (B * seq_len, opt.i_f_len)

        return out.view(batch_size, seq_len, -1)
        # return: (B, seq_len, opt.i_f_len)


class LightSensitive_Module(nn.Module):
    def __init__(self, opt):
        super(LightSensitive_Module, self).__init__()
        # CNN
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)

    def forward(self, img):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        # image CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image(v)
        v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v = self.visual_head(v)  # (batch, seq_len, 256)

        return v

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


class MuscleDriving_Module(nn.Module):
    def __init__(self, opt):
        super(MuscleDriving_Module, self).__init__()
        self.seq_len = opt.seq_len - 1
        self.encoder = Inertial_encoder(opt)

    def forward(self, imu):
        # imu : (B, seq_len * imu_len + 1, 6)
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(self.seq_len)], dim=1)
        # imu : (B, seq_len, imu_len, 6)
        fv = self.encoder(imu)

        return fv  # (B, seq_len, i_f_len)


class InstinctBias_Module(nn.Module):
    def __init__(self, opt):
        super(InstinctBias_Module, self).__init__()
        self.fuse_method = opt.fuse_method
        self.f_len = opt.i_f_len + opt.v_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len))
        elif self.fuse_method == 'rnn_IB':
            # rnn网络设置
            self.net = nn.LSTM(
                input_size=self.f_len,
                hidden_size=64,
                num_layers=4,
                dropout=opt.rnn_dropout_between,
                batch_first=True)
            # The output networks
            self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
            self.regressor = nn.Sequential(
                nn.Linear(64, 32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(32, 2))
            in_dim = opt.i_f_len + opt.v_f_len
            self.conf_net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(256),
                nn.Linear(256, 32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(32),
                nn.Linear(32, 2))
        elif self.fuse_method == 'ncp_IB':
            self.wiring = AutoNCP(units=opt.ncp_units, output_size=2)
            self.net = CfC(self.f_len, self.wiring)
            self.conf_net = nn.Sequential(
                nn.Linear(self.f_len, 256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(256),
                nn.Linear(256, 32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(32),
                nn.Linear(32, 2))
        
    def forward(self, v, i, temp, h0=None, random=False):
        # v: (B, 1, opt.v_f_len)
        # i: (B, 1, opt.i_f_len)
        bz = v.shape[0]
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1), None, torch.zeros(bz, 1, 2), torch.zeros(bz, 1, 2)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights, None, torch.zeros(bz, 1, 2), torch.zeros(bz, 1, 2)
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0], None, torch.zeros(bz, 1, 2), torch.zeros(bz, 1, 2)
        elif self.fuse_method == 'rnn_IB':
            if random:
                avi = torch.cat([v, i], dim=-1)

                decision = (torch.rand(bz, 1, 2) < 0.5).float()
                decision[:, :, 1] = 1 - decision[:, :, 0]
                decision = decision.to(v.device)
                # (B, 1, 2)

                selection = (torch.rand(bz, 1, 2) < 0.5).float()
                selection[:, :, 1] = 1 - selection[:, :, 0]
                selection = selection.to(v.device)
                # (B, 1, 2)

                v_w = v * selection[:, :, :1]
                i_w = i * selection[:, :, -1:]
                svi = torch.cat([v_w, i_w], dim=-1)

                h1 = None
            else:
                # Step 1: 将视觉和惯性模态特征 concat
                avi = torch.cat([v, i], dim=-1)

                # Step 2: 通过 net 获取 RNN 输出的 logits 和隐藏状态 h1
                if h0 is not None:
                    h0 = (h0[0].transpose(1, 0).contiguous(), h0[1].transpose(1, 0).contiguous())
                logit, h1 = self.net(avi.detach()) if h0 is None else self.net(avi.detach(), h0)
                logit = self.rnn_drop_out(logit)
                logit = self.regressor(logit.view(bz, -1))
                logit = logit.unsqueeze(1)
                h1 = (h1[0].transpose(1, 0).contiguous(), h1[1].transpose(1, 0).contiguous())
                
                # Step 3: 使用 Gumbel-Softmax 得到是否使用融合模态的决策 decision
                decision = F.gumbel_softmax(logit, tau=temp, hard=True, dim=-1)  # (B, 1, 2)

                # Step 4: 通过置信度网络为每个模态生成置信度
                conf_logit = self.conf_net(avi.view(avi.shape[0], -1).detach())  # 输入融合特征到置信度网络
                conf_logit = conf_logit.unsqueeze(1)  # (B, 1, 2)

                # Step 5: 使用 Gumbel-Softmax 得到使用哪个模态的选择 selection
                selection = F.gumbel_softmax(conf_logit, tau=temp, hard=True, dim=-1)  # (B, 1, 2)

                # Step 6: 根据 selection 选择 v 或 i
                v_w = v * selection[:, :, :1]
                i_w = i * selection[:, :, -1:]
                svi = torch.cat([v_w, i_w], dim=-1)

            fusion_feature = avi * decision[:, :, :1] + svi * decision[:, :, -1:]  # (B, 1, f_len)
            # 最终输出
            output = torch.cat([fusion_feature], dim=-1)
            # output = torch.cat([fusion_feature, v_w, i_w], dim=-1)
            return output, h1, decision, selection[:, :, :1]  # 返回加权特征、隐藏状态和决策权重
        elif self.fuse_method == 'ncp_IB':
            if random:
                avi = torch.cat([v, i], dim=-1)

                decision = (torch.rand(bz, 1, 2) < 0.5).float()
                decision[:, :, 1] = 1 - decision[:, :, 0]
                decision = decision.to(v.device)
                # (B, 1, 2)

                selection = (torch.rand(bz, 1, 2) < 0.5).float()
                selection[:, :, 1] = 1 - selection[:, :, 0]
                selection = selection.to(v.device)
                # (B, 1, 2)

                v_w = v * selection[:, :, :1]
                i_w = i * selection[:, :, -1:]
                svi = torch.cat([v_w, i_w], dim=-1)

                h1 = None
            else:
                # Step 1: 将视觉和惯性模态特征 concat
                avi = torch.cat([v, i], dim=-1)

                # Step 2: 通过 net 获取 RNN 输出的 logits 和隐藏状态 h1
                logit, h1 = self.net(avi.detach()) if h0 is None else self.net(avi.detach(), h0)

                # Step 3: 使用 Gumbel-Softmax 得到是否使用融合模态的决策 decision
                decision = F.gumbel_softmax(logit, tau=temp, hard=True, dim=-1)  # (B, 1, 2)

                # Step 4: 通过置信度网络为每个模态生成置信度
                conf_logit = self.conf_net(avi.view(avi.shape[0], -1).detach())  # 输入融合特征到置信度网络
                # xin = torch.cat([avi, logit], dim=-1).view(avi.shape[0],-1)  # (B, 2)
                # conf_logit = self.conf_net(xin.detach())  # 输入融合特征到置信度网络
                conf_logit = conf_logit.unsqueeze(1)  # (B, 1, 2)

                # Step 5: 使用 Gumbel-Softmax 得到使用哪个模态的选择 selection
                selection = F.gumbel_softmax(conf_logit, tau=temp, hard=True, dim=-1)  # (B, 1, 2)

                # Step 6: 根据 selection 选择 v 或 i
                v_w = v * selection[:, :, :1]
                i_w = i * selection[:, :, -1:]
                svi = torch.cat([v_w, i_w], dim=-1)

            fusion_feature = avi * decision[:, :, :1] + svi * decision[:, :, -1:]  # (B, 1, f_len)
            # 最终输出
            output = torch.cat([fusion_feature], dim=-1)
            # output = torch.cat([fusion_feature, v_w, i_w], dim=-1)
            return output, h1, decision, selection[:, :, :1]  # 返回加权特征、隐藏状态和决策权重


class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()

        # The main RNN network
        f_len = opt.v_f_len + opt.i_f_len
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, fused, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())

        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc


class WormVIO(nn.Module):
    def __init__(self, opt):
        super(WormVIO, self).__init__()

        self.timer = Timer()
        self.LS = LightSensitive_Module(opt)
        self.MD = MuscleDriving_Module(opt)
        self.IB = InstinctBias_Module(opt)
        self.Pose_net = Pose_RNN(opt)

        initialization(self)

    def forward(self, img, imu, is_first=True, hc=None, ncp_hc=None, temp=5, random=False):
        # img : (B, seq_len+1, 3, H, W)
        # imu : (B, seq_len * imu_len + 1, 6)
        self.timer.tic('visual')
        fv = self.LS(img)
        self.timer.toc('visual')
        self.timer.tic('inertial')
        fi = self.MD(imu)
        self.timer.toc('inertial')

        # batch_size = fv.shape[0]
        seq_len = fv.shape[1]
        poses = []
        decisions = []
        selections = []
        self.timer.tic('pose&IB')
        for i in range(seq_len):
            if i == 0 and is_first:
                pose, hc = self.Pose_net(torch.cat([fv[:, i:i + 1, :], fi[:, i:i + 1, :]], dim=-1), hc)
            else:
                fused, ncp_hc, decision, selection = self.IB(fv[:, i:i + 1, :], fi[:, i:i + 1, :], temp, ncp_hc, random)
                pose, hc = self.Pose_net(fused, hc)
                decisions.append(decision)
                selections.append(selection)
            poses.append(pose)
        poses = torch.cat(poses, dim=1)
        decisions = torch.cat(decisions, dim=1)
        selections = torch.cat(selections, dim=1)
        self.timer.toc('pose&IB')
        return poses, hc, ncp_hc, decisions, selections


def initialization(net):
    # Initialization
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m,
                                                                                                                   nn.Linear):
            # 使用 Kaiming 正态初始化来初始化卷积层和全连接层的权重
            nn.init.kaiming_normal_(m.weight.data)
            # nn.init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            # 对批归一化层的权重初始化为1，偏置初始化为0
            m.weight.data.fill_(1)
            m.bias.data.zero_()
