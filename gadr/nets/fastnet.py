import torch.nn as nn
import torch.nn.functional as F
from nets.feature import GANetFeature,FeaturePyrmaid

# from nets.model import FeaturePyrmaid, FeaturePyramidNetwork
# from nets.resnet import FastNetFeature



from nets.refinement import  StereoDRNetRefinement

from nets.submodule import *
from nets.threeD_aggregation import hourglass2



class FastNet(nn.Module):
    def __init__(self, max_disp,
                 num_downsample=2,
                 no_feature_mdconv=False,
                 feature_pyramid=True,
                 feature_pyramid_network=True,
                 feature_similarity='correlation',
                 num_scales=3,
                 refinement_type='stereodrnet',
                 no_intermediate_supervision=False,
                 num_stage_blocks=1):
        super(FastNet, self).__init__()

        self.refinement_type = refinement_type
        self.feature_pyramid = feature_pyramid
        self.feature_pyramid_network = feature_pyramid_network
        self.num_downsample = num_downsample
        self.num_scales = num_scales



        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU())



        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
        self.corr_stem = nn.Sequential(
            nn.Conv3d(8,8,kernel_size=(1,3,3),padding=(0,4,4),stride=1,dilation=4),
            BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1,dilation=1))
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass2(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)








#feature extractor)
        self.feature_extractor = GANetFeature()
        self.max_disp = max_disp // 4
        self.fpn = FeaturePyrmaid()


# Refinement

        # refine_module_list = nn.ModuleList()
        #
        # refine_module_list.append(StereoDRNetRefinement())
        #
        # self.refinement = refine_module_list

        self.refinement = StereoDRNetRefinement()


    def feature_extraction(self, img):
        feature = self.feature_extractor(img)
        feature = self.fpn(feature)
        return feature



    def cost_volume_construction(self, left_feature, right_feature):
        cost_volume = self.cost_volume(left_feature, right_feature)

        return cost_volume

    def disparity_refinement(self, left_img, right_img,disparity):

        scale_factor = 0.5


        curr_left_img = F.interpolate(left_img,
                                      scale_factor=scale_factor,
                                      mode='bilinear', align_corners=False,recompute_scale_factor=True)
        curr_right_img = F.interpolate(right_img,
                                       scale_factor=scale_factor,
                                       mode='bilinear', align_corners=False,recompute_scale_factor=True)
        inputs = (disparity, curr_left_img, curr_right_img)
        disparity_up = self.refinement(*inputs)

        return disparity_up

    def forward(self, left_img, right_img):
#fpn feature extrator
        left_feature = self.feature_extraction(left_img)
        right_feature = self.feature_extraction(right_img)
#build gwc correlation volume
        stem_2x = self.stem_2(left_img)
        stem_4x = self.stem_4(stem_2x)


        stem_2y = self.stem_2(right_img)
        stem_4y = self.stem_4(stem_2y)


        left_feature[0] = torch.cat((left_feature[0], stem_4x), 1)
        right_feature[0] = torch.cat((right_feature[0], stem_4y), 1)
        match_left = self.desc(self.conv(left_feature[0]))
        match_right = self.desc(self.conv(right_feature[0]))
        gwc_volume = build_gwc_volume(match_left, match_right, 192 // 4,8)
        gwc_volume = self.corr_stem(gwc_volume)
        gwc_volume = self.corr_feature_att(gwc_volume, left_feature[0])

# 3D hourglass cost aggregation

        cost = self.cost_agg(gwc_volume, left_feature)

        xspx = self.spx_4(left_feature[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

#Init disp from geometry encoding volume
        prob = F.softmax(self.classifier(cost).squeeze(1), dim=1)
        initial_disp = disparity_regression(prob, self.max_disp)

        initial_disp = initial_disp.squeeze(1)
        disparity_pyramid = []
        disparity_pyramid.append(initial_disp)


        middle_disp = self.disparity_refinement(left_img,right_img,disparity_pyramid[0])
        disparity_pyramid.append(middle_disp)
        middle_disp = middle_disp.unsqueeze(1)
        final_disp = context_upsample(middle_disp*2.,spx_pred)

        disparity_pyramid.append(final_disp)





        return disparity_pyramid
