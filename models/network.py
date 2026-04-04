import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DoubleConv, Down, Up, init_weights
from .AdaptiveFrequencyReassemble import AdaptiveFrequencyReassemble
from .visual_feature import save_advanced_visualization
import random

class GradConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm='ins'):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False), nn.InstanceNorm3d(out_channels), nn.LeakyReLU(inplace=True))
        self.grad_conv = nn.Conv3d(in_channels, out_channels, 3, padding=1, groups=in_channels, bias=False)
        with torch.no_grad():
            self.grad_conv.weight.fill_(0); c=1; self.grad_conv.weight[:,:,c,c,c]=-6
            self.grad_conv.weight[:,:,c-1,c,c]=1; self.grad_conv.weight[:,:,c+1,c,c]=1
            self.grad_conv.weight[:,:,c,c-1,c]=1; self.grad_conv.weight[:,:,c,c+1,c]=1
            self.grad_conv.weight[:,:,c,c,c-1]=1; self.grad_conv.weight[:,:,c,c,c+1]=1
    def forward(self, x): return self.conv(x) + self.grad_conv(x)

class LargeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, 5, padding=2, bias=False), nn.InstanceNorm3d(out_channels), nn.LeakyReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, chs, is_high=True):
        super().__init__()
        self.level0 = GradConv(in_channels, chs[0]) if is_high else LargeConv(in_channels, chs[0])
        self.down1 = Down(chs[0], chs[1]); self.down2 = Down(chs[1], chs[2]); self.down3 = Down(chs[2], chs[3]); self.down4 = Down(chs[3], chs[4])
    def forward(self, x):
        x0 = self.level0(x); x1 = self.down1(x0); x2 = self.down2(x1); x3 = self.down3(x2); x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class CDFreqNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=5):
        super(CDFreqNet, self).__init__()
        self.chs = (16, 32, 64, 128, 256)
        
        self.high_encoder = SimpleEncoder(1, self.chs, True)
        self.low_encoder = SimpleEncoder(1, self.chs, False)
        
        bot_ch = self.chs[-1] 
        
        self.fusion_conv = nn.Conv3d(bot_ch * 2, bot_ch, kernel_size=1)
     
        self.afr_bottle = AdaptiveFrequencyReassemble(self.chs[4], num_tokens=16, freq_k=4)
        self.afr_skip3 = AdaptiveFrequencyReassemble(self.chs[3], num_tokens=16, freq_k=4)
        self.afr_skip2 = AdaptiveFrequencyReassemble(self.chs[2], num_tokens=8, freq_k=4)

        self.up1 = Up(self.chs[4] + self.chs[3], self.chs[3])
        self.up2 = Up(self.chs[3] + self.chs[2], self.chs[2])
        self.up3 = Up(self.chs[2] + self.chs[1] * 2, self.chs[1])
        self.up4 = Up(self.chs[1] + self.chs[0] * 2, self.chs[0])
        
        self.out_conv = nn.Conv3d(self.chs[0], num_classes, kernel_size=1)
        self.act = nn.Softmax(dim=1)
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d): torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear): torch.nn.init.kaiming_normal_(m.weight) 
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None: m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x_high, x_low, label=None, return_proto=False, mod='A', rmmax=40, epoch=1):

        high_feats = self.high_encoder(x_high)
        low_feats = self.low_encoder(x_low)
            
        if return_proto: return high_feats[-1]

        bottle = self.afr_bottle(high_feats[-1], low_feats[-1])

        skip3 = self.afr_skip3(high_feats[-2], low_feats[-2])
        x = self.up1(bottle, skip3) # 256 + 128 -> 128

        skip2 = self.afr_skip2(high_feats[-3], low_feats[-3])
        x = self.up2(x, skip2)

        skip1 = torch.cat([low_feats[-4], high_feats[-4]], dim=1)
        x = self.up3(x, skip1)

        skip0 = torch.cat([low_feats[0], high_feats[0]], dim=1)
        x_final = self.up4(x, skip0)
        
        return self.act(self.out_conv(x_final)), x_final







