import torch
import torch.nn as nn
from models.model_utils import *


class Pointnet2(nn.Module):
    """
    PointNet++
    """

    def __init__(self):
        super(Pointnet2, self).__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNet_SA_Module_KNN(
                npoint=8192,
                nsample=64,
                in_channel=3,
                mlp=[32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointNet_SA_Module_KNN(
                npoint=2048,
                nsample=64,
                in_channel=64,
                mlp=[64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointNet_SA_Module_KNN(
                npoint=512,
                nsample=32,
                in_channel=128,
                mlp=[128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointNet_SA_Module_KNN(
                npoint=256,
                nsample=32,
                in_channel=256,
                mlp=[256, 256, 512],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointNet_SA_Module_KNN(
                npoint=64,
                nsample=16,
                in_channel=512,
                mlp=[512, 512, 1024],
                use_xyz=True,
                group_all=True
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointNet_FP_Module(in_channel=128, mlp=[128, 128, 128], use_points1=True, in_channel_points1=3)
        )
        self.FP_modules.append(
            PointNet_FP_Module(in_channel=256, mlp=[256, 128], use_points1=True, in_channel_points1=64))
        self.FP_modules.append(
            PointNet_FP_Module(in_channel=256, mlp=[256, 256], use_points1=True, in_channel_points1=128))
        self.FP_modules.append(
            PointNet_FP_Module(in_channel=512, mlp=[256, 256], use_points1=True, in_channel_points1=256))
        self.FP_modules.append(
            PointNet_FP_Module(in_channel=1024, mlp=[512, 512], use_points1=True, in_channel_points1=512))

        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1)

    def forward(self, pc):
        """

        :param pc: input point cloud [B, N, 3]
        :return:
        """
        xyz = pc[..., 0:3].transpose(1, 2).contiguous()  # [B, 3, N]
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else xyz  # [B, C, N]
        l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        feat = F.relu(self.bn(self.conv1(l_features[0])))
        x = self.drop(feat)
        x = self.conv2(x).transpose(1, 2).contiguous()  # [B, N, 1]
        x = F.sigmoid(x)

        return x.squeeze(-1)


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        """

        :param pred: [B, N]
        :param target: [B, N]
        :return:
        """
        return F.mse_loss(pred, target)


def get_model():
    model = Pointnet2()
    return model


if __name__ == '__main__':
    print('hello world!')
    batch_size = 8

    # model test
    model = get_model().cuda()
    pc = torch.rand(batch_size, 16384, 3).cuda()
    lm_pre = model(pc)
    print(lm_pre.shape)
    print('model runable')

    # loss test
    loss = get_loss().cuda()
    lm_gt = torch.rand(batch_size, 16384).cuda()
    loss_value = loss(lm_pre, lm_gt)
    print('loss runable')
