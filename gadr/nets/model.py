import torch
import torch.nn as nn
import torch.nn.functional as F






class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=128,
                 num_levels=3):
        # FPN paper uses 256 out channels by default
        super(FeaturePyramidNetwork, self).__init__()

        assert isinstance(in_channels, list)

        self.in_channels = in_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(num_levels):
            lateral_conv = nn.Conv2d(in_channels[i], out_channels, 1)
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # Inputs: resolution high -> low
        assert len(self.in_channels) == len(inputs)

        # Build laterals
        laterals = [lateral_conv(inputs[i])
                    for i, lateral_conv in enumerate(self.lateral_convs)]

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # Build outputs
        out = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return out

#build correlation volume

class CostVolume(nn.Module):
    def __init__(self, max_disp, feature_similarity='correlation'):
        """Construct cost volume based on different
        similarity measures

        Args:
            max_disp: max disparity candidate
            feature_similarity: type of similarity measure
        """
        super(CostVolume, self).__init__()

        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.size()


        if self.feature_similarity == 'correlation':
            cost_volume = left_feature.new_zeros(b, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] *
                                                right_feature[:, :, :, :-i]).mean(dim=1)####dim=1:按行对应元素取均值
                else:
                    cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)

        else:
            raise NotImplementedError

        cost_volume = cost_volume.contiguous()  #[B, D, H, W]

        return cost_volume



class CostVolumePyramid(nn.Module):
    def __init__(self, max_disp, feature_similarity='correlation'):
        super(CostVolumePyramid, self).__init__()
        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

        self.patch_s1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=1, groups=1, padding=(1, 1), bias=False)
        self.patch_s2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, dilation=2,
                                  groups=1,padding=(2, 2), bias=False)
        self.patch_s3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, dilation=3,
                                  groups=1,padding=(3, 3), bias=False)

    def forward(self, left_feature_pyramid, right_feature_pyramid):
        num_scales = len(left_feature_pyramid)

        cost_volume_pyramid = []
        for s in range(num_scales):
            max_disp = self.max_disp // (2 ** s)
            cost_volume_module = CostVolume(max_disp, self.feature_similarity)
            cost_volume = cost_volume_module(left_feature_pyramid[s],right_feature_pyramid[s])
            if s==0:
                cost_volume=self.patch_s3(cost_volume)
            elif s==1:
                cost_volume = self.patch_s2(cost_volume)
            elif s==2:
                cost_volume = self.patch_s1(cost_volume)

            cost_volume_pyramid.append(cost_volume)

        return cost_volume_pyramid  # H/3, H/6, H/12