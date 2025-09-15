import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from rock_utils import STN3d, STNkd, feature_transform_reguliarzer 


class get_model(nn.Module):
    def __init__(self, part_num=3, color_channel=True):  
        super(get_model, self).__init__()
        if color_channel:
            channel = 6 
        else:
            channel = 3  
        self.part_num = part_num  
        self.color_channel = color_channel  
        self.stn = STN3d(channel)  
        
        # 特征提取卷积层
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        
        # Batch Normalization层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        
        # 特征空间变换网络
        self.fstn = STNkd(k=128)
        
        # 分割头卷积层（输入通道调整：2048+1(类别数) + 64+128+128+512+2048 = 4945）
        self.convs1 = torch.nn.Conv1d(4945, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1) 
        
        # 分割头的Batch Normalization层
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()  # B:批次大小, D:特征维度, N:点数量
        
        # 应用空间变换网络
        trans = self.stn(point_cloud)
        point_cloud_transposed = point_cloud.transpose(2, 1)  # [B, N, D]
        
        # 如果包含颜色信息，分离坐标和颜色特征
        if D > 3:
            point_cloud_xyz, features = point_cloud_transposed.split(3, dim=2)  # 分离xyz和rgb
        else:
            point_cloud_xyz = point_cloud_transposed
            features = None
        
        # 应用空间变换
        point_cloud_transformed = torch.bmm(point_cloud_xyz, trans)  # [B, N, 3]
        
        # 重新拼接颜色特征
        if D > 3:
            point_cloud_transformed = torch.cat([point_cloud_transformed, features], dim=2)
        
        # 转换回 [B, D, N] 格式用于卷积
        point_cloud = point_cloud_transformed.transpose(2, 1)  # [B, D, N]
        
        # 特征提取
        out1 = F.relu(self.bn1(self.conv1(point_cloud)))  # [B, 64, N]
        out2 = F.relu(self.bn2(self.conv2(out1)))         # [B, 128, N]
        out3 = F.relu(self.bn3(self.conv3(out2)))         # [B, 128, N]
        
        # 特征空间变换
        trans_feat = self.fstn(out3)  # [B, 128, 128]
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)  # [B, 128, N]
        
        # 更高层次特征提取
        out4 = F.relu(self.bn4(self.conv4(net_transformed)))  # [B, 512, N]
        out5 = self.bn5(self.conv5(out4))                     # [B, 2048, N]
        
        # 全局特征提取
        out_max = torch.max(out5, 2, keepdim=True)[0]  # [B, 2048, 1]
        out_max = out_max.view(-1, 2048)               # [B, 2048]
        
        # 拼接全局特征和类别标签（仅1个类别：复杂岩体）
        out_max = torch.cat([out_max, label.squeeze(1)], 1)  # [B, 2048+1]
        
        # 特征扩展到每个点
        expand = out_max.view(-1, 2049, 1).repeat(1, 1, N)  # [B, 2049, N]
        
        # 拼接所有特征
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)  # [B, 4945, N]
        
        # 分割头
        net = F.relu(self.bns1(self.convs1(concat)))  # [B, 256, N]
        net = F.relu(self.bns2(self.convs2(net)))     # [B, 256, N]
        net = F.relu(self.bns3(self.convs3(net)))     # [B, 128, N]
        net = self.convs4(net)                        # [B, 3, N]
        
        # 调整输出格式并应用softmax
        net = net.transpose(2, 1).contiguous()  # [B, N, 3]
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)  # [B*N, 3]
        net = net.view(B, N, self.part_num)  # [B, N, 3]

        return net, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # 计算分类损失
        loss = F.nll_loss(pred, target)
        # 计算特征变换正则化损失
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        # 总损失
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
