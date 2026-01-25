import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

"""
HydraLA-Net for Diabetic Retinopathy Multi-Label Semantic Segmentation
Adapted from: Lesion Aware Network (LANet) for Segmentation Task

This version includes a Hydra Head that splits the final segmentation head into 4 independent 
class-specific prediction branches.
"""

class HydraSegHead(nn.Module):
    def __init__(self, in_ch = 256, hidden_ratio = 0.25):
        super().__init__()
        hidden = max(1, int(in_ch * hidden_ratio))

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(in_ch, hidden, kernel_size = 1, bias = True),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size = 1, bias = True),
            )
            for name in ["EX", "HE", "MA", "SE"]
        })

        for head in self.heads.values():
            nn.init.constant_(head[-1].bias, -2.0)

    def forward(self, x):
        logits = [head(x) for head in self.heads.values()]
        return torch.cat(logits, dim = 1)
    
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out      = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out      = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out      = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out+residual, inplace=True)


class ResNet(nn.Module):
    def __init__(self, n_channels=3):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes*4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        weights_path = (
            Path(__file__).resolve().parent.parent
            / "pretrained"
            / "resnet50-19c8e357.pth"
        )

        if weights_path.exists():
            self.load_state_dict(
                torch.load(
                    weights_path,
                    map_location="cpu",
                    weights_only=False
                ),
                strict=False
            )
        else:
            print(f"[INFO] ResNet weights not found at {weights_path}")


class Head(nn.Module):
    """
    HAM (Head Attention Module).
    """
    def __init__(self, in_channel):
        super(Head, self).__init__()

        self.conv0 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel, 512, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(in_channel, 256, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.initialize_weights()

    def forward(self, input):
        left = F.relu(self.bn0(self.conv0(input)), inplace=True)
        wb = self.conv1(input)
        w, b = wb[:,:256,:,:], wb[:,256:,:,:]
        mid = F.relu(w * left + b, inplace=True)

        mid = F.relu(self.bn2(self.conv2(mid)), inplace=True)
        down = input.mean(dim=(2,3), keepdim=True)
        down = F.relu(self.conv3(down), inplace=True)
        down = torch.sigmoid(self.conv4(down))

        return mid * down

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class FPB(nn.Module):
    """
    Feature-Preserve Block (FPB).
    """
    def __init__(self, in_channel):
        super(FPB, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0   = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.initialize_weights()

    def forward(self, fea):
        fea = F.relu(self.bn0(self.conv0(fea)), inplace=True)
        down = fea.mean(dim=(2,3), keepdim=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = torch.sigmoid(self.conv2(down))
        return down

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class FFB(nn.Module):
    """
    Feature Fusion Block (FFB).
    """
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(FFB, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(256)

        self.conv_d1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l  = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3   = nn.Conv2d(256*3, 256, kernel_size=3, stride=1, padding=1)
        self.bn3     = nn.BatchNorm2d(256)

        self.initialize_weights()

    def forward(self, f_enc, f_dec, f_p):
        f_enc = F.relu(self.bn0(self.conv0(f_enc)), inplace=True)
        down_2 = self.conv_d2(f_enc)
        z1 = F.relu(down_2 * f_p, inplace=True)

        down = F.relu(self.bn1(self.conv1(f_dec)), inplace=True)

        if down.size()[2:] != f_enc.size()[2:]:
            down = F.interpolate(down, size=f_enc.size()[2:], mode='bilinear')
            z2 = F.relu(f_p * down, inplace=True)
        else:
            z2 = F.relu(f_p * down, inplace=True)

        if f_dec.size()[2:] != f_enc.size()[2:]:
            down_1 = F.interpolate(f_dec, size=f_enc.size()[2:], mode='bilinear')
            z3 = F.relu(down_1 * f_p, inplace=True)
        else:
            z3 = F.relu(f_dec * f_p, inplace=True)

        out = torch.cat((z1, z2, z3), dim=1)
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class LAM(nn.Module):
    """
    LAM (Lesion-Aware Module).
    """
    def __init__(self, in_channels):
        super(LAM, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))

        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))
        self.conv1_2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv1_2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        inter_channels = in_channels
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,  bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))

        self.initialize_weights()

    def forward(self, x):
        x1 = self.conv1_1(x)

        x1_1 = self.conv2_1(x1)  
        x1_2 = self.conv2_2(x1)  
        x1 = self.conv3(F.relu(x1_1 + x1_2, inplace=True))

        x2 = self.conv1_2(x)

        x2_ca = x2.mean(dim=(2,3), keepdim=True)
        x2_ca = F.relu(self.conv1_2_1(x2_ca), inplace=True)
        x2_ca = torch.sigmoid(self.conv1_2_2(x2_ca))

        x = self.conv4(x1 * x2_ca)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

#
class HydraLANet(nn.Module):
    """
    LANet (Lesion-Aware Network) (overall)
    """
    def __init__(self, n_channels=3, snapshot=None):
        super(HydraLANet, self).__init__()
        self.snapshot = snapshot
        self.bkbone  = ResNet(n_channels=n_channels)

        self.fpb45 = FPB(2048)
        self.fpb35 = FPB(2048)
        self.fpb25 = FPB(2048)

        self.head = Head(2048)

        self.ffb45   = FFB(1024,  256, 256)
        self.ffb34   = FFB( 512,  256, 256)
        self.ffb23   = FFB( 256,  256, 256)

        self.lam5    = LAM(256)
        self.lam4    = LAM(256)
        self.lam3    = LAM(256)
        self.lam2    = LAM(256)

        self.hydra_head = HydraSegHead(
            in_ch = 256,
            hidden_ratio = 0.25
        )
        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5_ = self.bkbone(x)

        # FPB
        out4_a = self.fpb45(out5_)
        out3_a = self.fpb35(out5_)
        out2_a = self.fpb25(out5_)

        # HAM
        out5 = self.head(out5_)

        # Out
        out5 = self.lam5(out5)
        out4 = self.lam4(self.ffb45(out4, out5, out4_a))  
        out3 = self.lam3(self.ffb34(out3, out4, out3_a)) 
        out2 = self.lam2(self.ffb23(out2, out3, out2_a)) 

        logits = self.hydra_head(out2)

        logits = F.interpolate(
            logits,
            size = x.size()[2:],
            mode = "bilinear",
            align_corners = False
        )

        return torch.sigmoid(logits)


    def initialize(self):
        if self.snapshot:
            try:
                self.load_state_dict(torch.load(self.snapshot))
            except:
                print("Warning: please check the snapshot file:", self.snapshot)
                pass
        else:
            for n, m in self.named_children():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                        
if __name__ == '__main__':
    net = HydraLANet(snapshot=None)