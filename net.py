import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from matplotlib import pyplot as plt
from torch.nn.modules.pooling import AdaptiveAvgPool1d, AvgPool1d, MaxPool1d


class RIConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(RIConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=1), nn.BatchNorm1d(out_channels), nn.LeakyReLU(negative_slope=0.1))

    def forward(self, x):
        # x = F.pad(x, [0, self.kernel_size-1], mode='circular')
        x = F.pad(x, [0, self.kernel_size-1], mode='constant', value=0)
        out = self.conv(x)
        return out


class RIDowsampling(nn.Module):
    def __init__(self, ratio=2):
        super(RIDowsampling, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        y = x[:, :, list(range(0, x.shape[2], self.ratio))].unsqueeze(1)
        for i in range(1, self.ratio):
            index = list(range(i, x.shape[2], self.ratio))
            y = torch.cat([y, x[:, :, index].unsqueeze(1)], 1)
        norm = torch.norm(torch.norm(y, 1, 2), 1, 2)
        idx = torch.argmax(norm, 1)
        idx = idx.unsqueeze(1).expand(x.shape[0], self.ratio)
        id_matrix = torch.tensor([list(range(self.ratio))]).expand(
            x.shape[0], self.ratio).to(device=x.device)
        out = y[id_matrix == idx]
        return out


class RIAttention(nn.Module):
    def __init__(self, channels):
        super(RIAttention, self).__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels), nn.Sigmoid())

    def forward(self, x):
        x1 = torch.mean(x, 2)
        w = self.fc(x1)
        w = w.unsqueeze(2)
        out = w*x
        return out


class RINet_attention_cons_pad(nn.Module):
    def __init__(self):
        super(RINet_attention_cons_pad, self).__init__()
        self.conv1 = nn.Sequential(RIAttention(12), RIConv(in_channels=12, out_channels=12, kernel_size=3), RIAttention(
            12), RIConv(in_channels=12, out_channels=16, kernel_size=3), RIAttention(16))
        self.conv2 = nn.Sequential(RIDowsampling(3), RIConv(
            in_channels=16, out_channels=16, kernel_size=3), RIAttention(16))
        self.conv3 = nn.Sequential(RIDowsampling(3), RIConv(
            in_channels=16, out_channels=32, kernel_size=3), RIAttention(32))
        self.conv4 = nn.Sequential(RIDowsampling(2), RIConv(
            in_channels=32, out_channels=32, kernel_size=3), RIAttention(32))
        self.conv5 = nn.Sequential(RIDowsampling(2), RIConv(
            in_channels=32, out_channels=64, kernel_size=3), RIAttention(64))
        self.conv6 = nn.Sequential(RIDowsampling(2), RIConv(
            in_channels=64, out_channels=128, kernel_size=3), RIAttention(128))

        #seq的卷积
        self.conv1_seq = nn.Sequential(RIAttention(12), RIConv(in_channels=12, out_channels=12, kernel_size=3), RIAttention(
            12), RIConv(in_channels=12, out_channels=16, kernel_size=3), RIAttention(16))
        self.conv2_seq = nn.Sequential(RIDowsampling(3), RIConv(
            in_channels=16, out_channels=16, kernel_size=3), RIAttention(16))
        self.conv3_seq = nn.Sequential(RIDowsampling(3), RIConv(
            in_channels=16, out_channels=32, kernel_size=3), RIAttention(32))
        self.conv4_seq = nn.Sequential(RIDowsampling(2), RIConv(
            in_channels=32, out_channels=32, kernel_size=3), RIAttention(32))
        self.conv5_seq = nn.Sequential(RIDowsampling(2), RIConv(
            in_channels=32, out_channels=64, kernel_size=3), RIAttention(64))
        self.conv6_seq = nn.Sequential(RIDowsampling(2), RIConv(
            in_channels=64, out_channels=128, kernel_size=3), RIAttention(128))


        self.pool = AdaptiveAvgPool1d(1)
        self.linear = nn.Sequential(nn.Linear(in_features=288, out_features=128), nn.LeakyReLU(
            negative_slope=0.1), nn.Linear(in_features=128, out_features=1)) 

    def forward(self, seq,x,y_0,y_1,y_2,y_3,y_4,y_5,y_6,y_7):

        featurex = self.gen_feature_fuse(seq,x)

        featurey_0 = self.gen_feature(y_0)
        featurey_1 = self.gen_feature(y_1)
        featurey_2 = self.gen_feature(y_2)
        featurey_3 = self.gen_feature(y_3)
        featurey_4 = self.gen_feature(y_4)
        featurey_5 = self.gen_feature(y_5)
        featurey_6 = self.gen_feature(y_6)
        featurey_7 = self.gen_feature(y_7)

        out_0, diff_0 = self.gen_score(featurex, featurey_0)
        out_1, diff_1 = self.gen_score(featurex, featurey_1)
        out_2, diff_2 = self.gen_score(featurex, featurey_2)
        out_3, diff_3 = self.gen_score(featurex, featurey_3)
        out_4, diff_4 = self.gen_score(featurex, featurey_4)
        out_5, diff_5 = self.gen_score(featurex, featurey_5)
        out_6, diff_6 = self.gen_score(featurex, featurey_6)
        out_7, diff_7 = self.gen_score(featurex, featurey_7)

        out_cat=torch.cat((out_0.reshape(1,-1),out_1.reshape(1,-1),out_2.reshape(1,-1),out_3.reshape(1,-1),out_4.reshape(1,-1),out_5.reshape(1,-1),out_6.reshape(1,-1),out_7.reshape(1,-1)),0)
        diff_cat=torch.cat((diff_0.reshape(1,-1),diff_1.reshape(1,-1),diff_2.reshape(1,-1),diff_3.reshape(1,-1),diff_4.reshape(1,-1),diff_5.reshape(1,-1),diff_6.reshape(1,-1),diff_7.reshape(1,-1)),0)
        out=torch.max(out_cat,0)[0]
        out_cat_idx=torch.max(out_cat,0)[1]
        w_indices=torch.arange(0,out.shape[0])
        diff=diff_cat[out_cat_idx,w_indices]
        return out, diff, out_cat

    def gen_feature_fuse(self, seq,x):
        fxy = []
        xy1_seq=self.conv1_seq(seq)
        xy1 = self.conv1(x)+xy1_seq
        fxy.append(self.pool(xy1).view(x.shape[0], -1))
        xy2_seq = self.conv2_seq(xy1_seq)
        xy2 = self.conv2(xy1)+xy2_seq
        fxy.append(self.pool(xy2).view(x.shape[0], -1))
        xy3_seq = self.conv3_seq(xy2_seq)
        xy3 = self.conv3(xy2)+xy3_seq
        fxy.append(self.pool(xy3).view(x.shape[0], -1))
        xy4_seq = self.conv4_seq(xy3_seq)
        xy4 = self.conv4(xy3)+xy4_seq
        fxy.append(self.pool(xy4).view(x.shape[0], -1))
        xy5_seq = self.conv5_seq(xy4_seq)
        xy5 = self.conv5(xy4)+xy5_seq
        fxy.append(self.pool(xy5).view(x.shape[0], -1))
        xy6_seq = self.conv6_seq(xy5_seq)
        xy6 = self.conv6(xy5)+xy6_seq
        fxy.append(self.pool(xy6).view(x.shape[0], -1))
        featurexy = torch.cat(fxy, 1)
        return featurexy


    def gen_feature(self, xy):
        fxy = []
        xy1 = self.conv1(xy)
        fxy.append(self.pool(xy1).view(xy.shape[0], -1))
        xy2 = self.conv2(xy1)
        fxy.append(self.pool(xy2).view(xy.shape[0], -1))
        xy3 = self.conv3(xy2)
        fxy.append(self.pool(xy3).view(xy.shape[0], -1))
        xy4 = self.conv4(xy3)
        fxy.append(self.pool(xy4).view(xy.shape[0], -1))
        xy5 = self.conv5(xy4)
        fxy.append(self.pool(xy5).view(xy.shape[0], -1))
        xy6 = self.conv6(xy5)
        fxy.append(self.pool(xy6).view(xy.shape[0], -1))
        featurexy = torch.cat(fxy, 1)
        return featurexy

    def gen_score(self, fx, fy):
        diff = torch.abs(fx-fy)
        # print('diff',diff)
        # print('diff.shape',diff.shape)
        # print('torch.norm(diff, dim=1)',torch.norm(diff, dim=1))
        # print('torch.norm(diff, dim=1).shape',torch.norm(diff, dim=1).shape)
        # os._exit()
        out = self.linear(diff).view(-1)
        if not self.training:
            out = torch.sigmoid(out)
        # print('out.shape',out.shape)
        # print('out',out)
        # print('torch.norm(diff, dim=1).shape',torch.norm(diff, dim=1).shape)
        # print('torch.norm(diff, dim=1)',torch.norm(diff, dim=1))
        # os._exit()
        return out, torch.norm(diff, dim=1)

    def load(self, model_file):
        checkpoint = torch.load(model_file)
        self.load_state_dict(checkpoint['state_dict'])


class RIConv_cir_pad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(RIConv_cir_pad, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=1), nn.BatchNorm1d(out_channels), nn.LeakyReLU(negative_slope=0.1))

    def forward(self, x):
        x = F.pad(x, [0, self.kernel_size-1], mode='circular')
        # x = F.pad(x, [0, self.kernel_size-1], mode='constant', value=0)
        out = self.conv(x)
        return out


class RINet_attention_cir_pad(nn.Module):
    def __init__(self):
        super(RINet_attention_cir_pad, self).__init__()
        self.conv1 = nn.Sequential(RIAttention(12), RIConv_cir_pad(in_channels=12, out_channels=12, kernel_size=3), RIAttention(
            12), RIConv_cir_pad(in_channels=12, out_channels=16, kernel_size=3), RIAttention(16))
        self.conv2 = nn.Sequential(RIDowsampling(3), RIConv_cir_pad(
            in_channels=16, out_channels=16, kernel_size=3), RIAttention(16))
        self.conv3 = nn.Sequential(RIDowsampling(3), RIConv_cir_pad(
            in_channels=16, out_channels=32, kernel_size=3), RIAttention(32))
        self.conv4 = nn.Sequential(RIDowsampling(2), RIConv_cir_pad(
            in_channels=32, out_channels=32, kernel_size=3), RIAttention(32))
        self.conv5 = nn.Sequential(RIDowsampling(2), RIConv_cir_pad(
            in_channels=32, out_channels=64, kernel_size=3), RIAttention(64))
        self.conv6 = nn.Sequential(RIDowsampling(2), RIConv_cir_pad(
            in_channels=64, out_channels=128, kernel_size=3), RIAttention(128))

        #seq的卷积
        self.conv1_seq = nn.Sequential(RIAttention(12), RIConv_cir_pad(in_channels=12, out_channels=12, kernel_size=3), RIAttention(
            12), RIConv_cir_pad(in_channels=12, out_channels=16, kernel_size=3), RIAttention(16))
        self.conv2_seq = nn.Sequential(RIDowsampling(3), RIConv_cir_pad(
            in_channels=16, out_channels=16, kernel_size=3), RIAttention(16))
        self.conv3_seq = nn.Sequential(RIDowsampling(3), RIConv_cir_pad(
            in_channels=16, out_channels=32, kernel_size=3), RIAttention(32))
        self.conv4_seq = nn.Sequential(RIDowsampling(2), RIConv_cir_pad(
            in_channels=32, out_channels=32, kernel_size=3), RIAttention(32))
        self.conv5_seq = nn.Sequential(RIDowsampling(2), RIConv_cir_pad(
            in_channels=32, out_channels=64, kernel_size=3), RIAttention(64))
        self.conv6_seq = nn.Sequential(RIDowsampling(2), RIConv_cir_pad(
            in_channels=64, out_channels=128, kernel_size=3), RIAttention(128))


        self.pool = AdaptiveAvgPool1d(1)
        self.linear = nn.Sequential(nn.Linear(in_features=288, out_features=128), nn.LeakyReLU(
            negative_slope=0.1), nn.Linear(in_features=128, out_features=1)) 

    def forward(self, seq,x,y_0,y_1,y_2,y_3,y_4,y_5,y_6,y_7):

        featurex = self.gen_feature_fuse(seq,x)

        featurey_0 = self.gen_feature(y_0)
        featurey_1 = self.gen_feature(y_1)
        featurey_2 = self.gen_feature(y_2)
        featurey_3 = self.gen_feature(y_3)
        featurey_4 = self.gen_feature(y_4)
        featurey_5 = self.gen_feature(y_5)
        featurey_6 = self.gen_feature(y_6)
        featurey_7 = self.gen_feature(y_7)

        out_0, diff_0 = self.gen_score(featurex, featurey_0)
        out_1, diff_1 = self.gen_score(featurex, featurey_1)
        out_2, diff_2 = self.gen_score(featurex, featurey_2)
        out_3, diff_3 = self.gen_score(featurex, featurey_3)
        out_4, diff_4 = self.gen_score(featurex, featurey_4)
        out_5, diff_5 = self.gen_score(featurex, featurey_5)
        out_6, diff_6 = self.gen_score(featurex, featurey_6)
        out_7, diff_7 = self.gen_score(featurex, featurey_7)

        out_cat=torch.cat((out_0.reshape(1,-1),out_1.reshape(1,-1),out_2.reshape(1,-1),out_3.reshape(1,-1),out_4.reshape(1,-1),out_5.reshape(1,-1),out_6.reshape(1,-1),out_7.reshape(1,-1)),0)
        diff_cat=torch.cat((diff_0.reshape(1,-1),diff_1.reshape(1,-1),diff_2.reshape(1,-1),diff_3.reshape(1,-1),diff_4.reshape(1,-1),diff_5.reshape(1,-1),diff_6.reshape(1,-1),diff_7.reshape(1,-1)),0)
        out=torch.max(out_cat,0)[0]
        out_cat_idx=torch.max(out_cat,0)[1]
        w_indices=torch.arange(0,out.shape[0])
        diff=diff_cat[out_cat_idx,w_indices]
        return out, diff, out_cat

    def gen_feature_fuse(self, seq,x):
        fxy = []
        xy1_seq=self.conv1_seq(seq)
        xy1 = self.conv1(x)+xy1_seq
        fxy.append(self.pool(xy1).view(x.shape[0], -1))
        xy2_seq = self.conv2_seq(xy1_seq)
        xy2 = self.conv2(xy1)+xy2_seq
        fxy.append(self.pool(xy2).view(x.shape[0], -1))
        xy3_seq = self.conv3_seq(xy2_seq)
        xy3 = self.conv3(xy2)+xy3_seq
        fxy.append(self.pool(xy3).view(x.shape[0], -1))
        xy4_seq = self.conv4_seq(xy3_seq)
        xy4 = self.conv4(xy3)+xy4_seq
        fxy.append(self.pool(xy4).view(x.shape[0], -1))
        xy5_seq = self.conv5_seq(xy4_seq)
        xy5 = self.conv5(xy4)+xy5_seq
        fxy.append(self.pool(xy5).view(x.shape[0], -1))
        xy6_seq = self.conv6_seq(xy5_seq)
        xy6 = self.conv6(xy5)+xy6_seq
        fxy.append(self.pool(xy6).view(x.shape[0], -1))
        featurexy = torch.cat(fxy, 1)
        return featurexy


    def gen_feature(self, xy):
        fxy = []
        xy1 = self.conv1(xy)
        fxy.append(self.pool(xy1).view(xy.shape[0], -1))
        xy2 = self.conv2(xy1)
        fxy.append(self.pool(xy2).view(xy.shape[0], -1))
        xy3 = self.conv3(xy2)
        fxy.append(self.pool(xy3).view(xy.shape[0], -1))
        xy4 = self.conv4(xy3)
        fxy.append(self.pool(xy4).view(xy.shape[0], -1))
        xy5 = self.conv5(xy4)
        fxy.append(self.pool(xy5).view(xy.shape[0], -1))
        xy6 = self.conv6(xy5)
        fxy.append(self.pool(xy6).view(xy.shape[0], -1))
        featurexy = torch.cat(fxy, 1)
        return featurexy

    def gen_score(self, fx, fy):
        diff = torch.abs(fx-fy)
        # print('diff',diff)
        # print('diff.shape',diff.shape)
        # print('torch.norm(diff, dim=1)',torch.norm(diff, dim=1))
        # print('torch.norm(diff, dim=1).shape',torch.norm(diff, dim=1).shape)
        # os._exit()
        out = self.linear(diff).view(-1)
        if not self.training:
            out = torch.sigmoid(out)
        # print('out.shape',out.shape)
        # print('out',out)
        # print('torch.norm(diff, dim=1).shape',torch.norm(diff, dim=1).shape)
        # print('torch.norm(diff, dim=1)',torch.norm(diff, dim=1))
        # os._exit()
        return out, torch.norm(diff, dim=1)

    def load(self, model_file):
        checkpoint = torch.load(model_file)
        self.load_state_dict(checkpoint['state_dict'])



if __name__ == "__main__":
    net = RINet_attention_cons_pad()
    net.eval()
    a = np.random.random(size=[32, 12, 360])
    b = np.random.random(size=[32, 12, 360])
    c = np.roll(b, random.randint(1, 360), 2)
    a = torch.from_numpy(np.array(a, dtype='float32'))
    b = torch.from_numpy(np.array(b, dtype='float32'))
    c = torch.from_numpy(np.array(c, dtype='float32'))
    # out1,_=net(a,c)
    # out2,_=net(a,b)
    out3, diff = net(c, b)
    print(diff)
    print('net orignal version script')
    # print(norm.shape)
    # print(out1)
    # print(out2)
    # print(out3)
