import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class HighFrequencyEnhance(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.dwconv = nn.Conv2d(in_ch, in_ch, 5, padding=2, groups=in_ch)
        self.bn = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        residual = x
        # 自定义拉普拉斯边缘检测
        laplacian_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                                        dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        x_edge = F.conv2d(x.mean(dim=1, keepdim=True), laplacian_kernel, padding=1)
        x = self.dwconv(x + x_edge * 0.3)
        return self.bn(x) + residual


class DRC(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.offset_conv = nn.Conv2d(channels, 18, 3, padding=1)
        self.deform_conv = DeformConv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        offsets = self.offset_conv(x)
        x_deform = self.deform_conv(x, offsets)
        return x * torch.sigmoid(x_deform)

class ASPP_Improved(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[6, 12, 18]):
        super().__init__()
        # 1. 统一中间通道数为in_channels//4减少计算量
        mid_channels = max(in_channels // 4, 64)

        # 2. 增加各分支梯度缩放系数
        self.branch_weights = nn.Parameter(torch.ones(2 + len(dilation_rates)))

        # 1x1卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 空洞卷积分支
        self.atrous_convs = nn.ModuleList()
        for rate in dilation_rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                # 增加局部梯度归一化
                nn.Dropout2d(0.1)  # 新增正则化
            ))

        # 全局池化分支
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)  # 新增正则化
        )

        # 融合层改进
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * (2 + len(dilation_rates)), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 增加更强的正则化
            nn.Dropout2d(0.5)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 分层初始化：不同层使用不同初始化增益
                if m in self.atrous_convs:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        branches = []
        # 1x1分支
        branches.append(self.conv1x1(x))

        # 空洞卷积分支
        for conv in self.atrous_convs:
            branches.append(conv(x))

        # 全局池化分支
        gap = self.gap(x)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        branches.append(gap)

        # 加权融合
        weighted_branches = [w * branch for w, branch in zip(self.branch_weights, branches)]
        return self.fusion(torch.cat(weighted_branches, dim=1))


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        # 1x1 卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 多尺度空洞卷积分支
        self.conv3x3 = nn.ModuleList()
        for rate in dilation_rates:
            self.conv3x3.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        # 全局平均池化分支
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 融合后的 1x1 卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(dilation_rates)), in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        features = []
        features.append(self.conv1x1(x))
        for conv in self.conv3x3:
            features.append(conv(x))
        gap = self.gap(x)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        features.append(gap)
        return self.fusion(torch.cat(features, dim=1))

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'resnet50'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.hfeat1 = HighFrequencyEnhance(64)  # resnet50的stage1输出通道
        self.hfeat2 = HighFrequencyEnhance(256)  # stage2输出通道
        self.hfeat3 = HighFrequencyEnhance(512)  # stage3
        self.hfeat4 = HighFrequencyEnhance(1024)  # stage4

        self.drc = DRC(2048)

        self.aspp = ASPP_Improved(2048, 2048)

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)

            print(feat1.shape)
        elif self.backbone == "resnet50":
            # print(44)
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
            # print(feat1.shape)
            # print(feat2.shape)
            # print(feat3.shape)
            #
            # print(feat4.shape)
            #
            # print(feat5.shape)



        feats1 = self.hfeat1(feat1)  # 处理stage1特征
        feats2 = self.hfeat2(feat2)  # stage2
        feats3 = self.hfeat3(feat3)
        feats4 = self.hfeat4(feat4)

        feats5 = self.aspp(feat5)
        feats5 = self.drc(feats5)

        up4 = self.up_concat4(feats4, feats5)
        up3 = self.up_concat3(feats3, up4)
        up2 = self.up_concat2(feats2, up3)
        up1 = self.up_concat1(feats1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


# net = Unet(2, True, 'resnet50')
# X = torch.rand(size=(4, 2, 512, 512), dtype=torch.float32)
# for layer in net.children():
#     print(layer)