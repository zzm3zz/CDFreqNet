import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DoubleConv, Down, Up, init_weights
from .AdaptiveFrequencyReassemble import AdaptiveFrequencyReassemble
from .visual_feature import save_advanced_visualization
import random

# ==============================================================================
# 1. 基础 Encoder 组件 
# ==============================================================================
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
    def __init__(self, in_channels, chs, is_structure=True):
        super().__init__()
        self.level0 = GradConv(in_channels, chs[0]) if is_structure else LargeConv(in_channels, chs[0])
        self.down1 = Down(chs[0], chs[1]); self.down2 = Down(chs[1], chs[2]); self.down3 = Down(chs[2], chs[3]); self.down4 = Down(chs[3], chs[4])
    def forward(self, x):
        x0 = self.level0(x); x1 = self.down1(x0); x2 = self.down2(x1); x3 = self.down3(x2); x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


# ==============================================================================
# 3. 主网络
# ==============================================================================
class CDFreqNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=5):
        super(CDFreqNet, self).__init__()
        self.chs = (16, 32, 64, 128, 256)
        
        self.struct_encoder = SimpleEncoder(1, self.chs, True)
        self.style_encoder = SimpleEncoder(1, self.chs, False)
        
        bot_ch = self.chs[-1] # 256

        # 瓶颈层融合
        self.fusion_conv = nn.Conv3d(bot_ch * 2, bot_ch, kernel_size=1)
        # 跳跃连接融合
        self.afr_bottle = AdaptiveFrequencyReassemble(self.chs[4], num_tokens=16, freq_k=4)
        self.afr_skip3 = AdaptiveFrequencyReassemble(self.chs[3], num_tokens=16, freq_k=4)
        self.afr_skip2 = AdaptiveFrequencyReassemble(self.chs[2], num_tokens=8, freq_k=4)

        # 3. [修改] Up 模块输入通道调整
        # 原来是 self.chs[3]*2，因为简单的 concat 是双倍通道。
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
            elif isinstance(m, nn.Linear): torch.nn.init.kaiming_normal_(m.weight) # 初始化 FC 层
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None: m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x_struct, x_style, label=None, return_proto=False, mod='A', rmmax=40, epoch=1):

        # --- 1. Encoding ---
        struct_feats = self.struct_encoder(x_struct)
        style_feats = self.style_encoder(x_style)
            
        if return_proto: return struct_feats[-1]
        
        vis_path = '/public/home/zhangzengmin/CDFreqNet/visual/whs_feature_layer/'
        vis_prob = 0.000001
        if vis_path is not None and vis_prob > 0 and random.random() < vis_prob:
            save_advanced_visualization(
                x_struct, x_style, label,
                struct_feats, style_feats,
                str(epoch)+mod, 0, vis_path
            )
            
        # --- 2. Bottleneck Fusion ---
        bottle = self.afr_bottle(struct_feats[-1], style_feats[-1])
        
        # Layer 3
        # 128 LF + 128 HF -> 128 Fused
        skip3 = self.afr_skip3(struct_feats[-2], style_feats[-2])
        x = self.up1(bottle, skip3) # 256 + 128 -> 128
        
        # Layer 2
        skip2 = self.afr_skip2(struct_feats[-3], style_feats[-3])
        x = self.up2(x, skip2)

        # Layer 1
        skip1 = torch.cat([style_feats[-4], struct_feats[-4]], dim=1)
        x = self.up3(x, skip1)

        # Layer 0
        skip0 = torch.cat([style_feats[0], struct_feats[0]], dim=1)
        x_final = self.up4(x, skip0)
        
        return self.act(self.out_conv(x_final)), x_final


class UNet_base(nn.Module):
    def __init__(self, input_channels=1, chs=(16, 32, 64, 128, 64, 32, 16), num_classes=5, is_batch=False):
        super(UNet_base, self).__init__()
        if is_batch:
            self.norm = "batch"
        else:
            self.norm = "ins"
        self.inc = DoubleConv(input_channels, chs[0]) 
        self.down1 = Down(chs[0], chs[1], self.norm)
        self.down2 = Down(chs[1], chs[2], self.norm)
        self.down3 = Down(chs[2], chs[3], self.norm)
        self.up1 = Up(chs[3] + chs[2], chs[4], self.norm)
        self.up2 = Up(chs[4] + chs[1], chs[5], self.norm)
        self.up3 = Up(chs[5] + chs[0], chs[6], self.norm)
        self.act = torch.nn.Softmax(1)
        self.out_conv = nn.Conv3d(chs[0], num_classes, kernel_size=1)
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.InstanceNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        Z = x.size()[2]
        Y = x.size()[3]
        X = x.size()[4]
        # diffZ = (16 - Z % 16) % 16
        # diffY = (16 - Y % 16) % 16
        # diffX = (16 - X % 16) % 16
        # x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # out = x[:, :, diffZ // 2: Z + diffZ // 2, diffY // 2: Y + diffY // 2, diffX // 2:X + diffX // 2]
        return self.act(self.out_conv(x))





