import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AttentionDeeLabv3p(nn.Module):
    def __init__(self):
        super(AttentionDeeLabv3p, self).__init__()
        self.backbone = Backbone()
        self.aspp = ASPP(dim_in=512)  # dim_in 512 for res18
        self.decoder = Decoder()

    def forward(self, x):
        x, low_feat = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, low_feat)
        return x

class Decoder(nn.Module):
    def __init__(self, num_classes=2):
        super(Decoder, self).__init__()
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(64, 48, 1), nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, 2, 1, stride=1)

    def forward(self, x, low_feat):
        # x: (32, 256, 8, 8), low_feat 32, 64, 64, 64
        low_feat = self.shortcut_conv(low_feat)
        x = F.interpolate(
            x,
            size=(low_feat.size(2), low_feat.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        x = self.cat_conv(torch.cat((x, low_feat), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=True)
        return x

class Backbone(nn.Module):
    """
    backbone using resnet18
    """

    def __init__(self):
        super(Backbone, self).__init__()
        self.resnet = models.resnet18(weights="IMAGENET1K_V1")

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        low_level_features = self.resnet.layer1(x)
        x = self.resnet.layer2(low_level_features)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x, low_level_features

class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out=256, dilations=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=dilations[0],
            ),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=3,
                stride=1,
                padding=dilations[1],
                dilation=dilations[1],
            ),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=3,
                stride=1,
                padding=dilations[2],
                dilation=dilations[2],
            ),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=3,
                stride=1,
                padding=dilations[3],
                dilation=dilations[3],
            ),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )
        self.se1 = SEModule(256)
        self.se2 = SEModule(256)
        self.se3 = SEModule(256)
        self.se4 = SEModule(256)
        self.se5 = SEModule(256)
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=256,
            kernel_size=(5, 1, 1),
            stride=(5, 1, 1),
            bias=False,
        )

    def forward(self, x):
        # x: [32, 512, 8, 8]
        # 1x1 conv
        x1 = self.b1(x)  # [32, 256, 8, 8]
        # atrous convolutions  # [32, 256, 8, 8]
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        # averge pooling
        x5 = self.avg_pool(x)  # [32, 512, 1, 1]
        x5 = self.b5(x5)  # [32, 256, 1, 1]
        x5 = F.interpolate(
            x5, size=x4.shape[2:], mode="bilinear", align_corners=False
        )  # [32, 256, 8, 8]

        # apply attention
        x1 = self.se1(x1)
        x2 = self.se2(x2)
        x3 = self.se3(x3)
        x4 = self.se4(x4)
        x5 = self.se5(x5)
        # apply attention
        # x = self.conv_bn_relu(x) # [32, 256, 8, 8] # origin

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = x.unsqueeze(1)  #  32, 1, 256, 8, 8

        x = self.conv3d(x).sum(dim=1)
        x = F.sigmoid(x)

        return x

class SEModule(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute global average pooling over spatial dimensions
        out = self.avg_pool(x).view(batch_size, channels)

        # Apply two fully-connected layers with ReLU activation and sigmoid activation
        out = self.fc2(self.relu(self.fc1(out)))
        out = self.sigmoid(out).view(batch_size, channels, 1, 1)

        # Multiply the input feature maps by the SE attention scores
        out = x * out.expand_as(x)

        return out
