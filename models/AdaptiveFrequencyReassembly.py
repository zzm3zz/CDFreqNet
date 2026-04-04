import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

class DecoupledGatedFusion(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.shared_fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True)
        )     
        self.gate_lf = nn.Sequential(
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() 
        )
        self.gate_hf = nn.Sequential(
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_lf, x_hf):
        b, c, _, _, _ = x_lf.size()

        combined = torch.cat([x_lf, x_hf], dim=1) # [B, 2C, D, H, W]
        context = self.avg_pool(combined).view(b, c * 2)

        shared_feat = self.shared_fc(context) # [B, C/r]
        
        w_lf = self.gate_lf(shared_feat).view(b, c, 1, 1, 1) * 2.0
        w_hf = self.gate_hf(shared_feat).view(b, c, 1, 1, 1) * 2.0

        out = w_lf * x_lf + w_hf * x_hf
        
        return out

class AdaptiveFrequencyReassemble(nn.Module):
 
    def __init__(self, channels, num_tokens=8, freq_k=4, scale_init=0.001):
        super().__init__()
        self.channels = channels
        self.num_tokens = num_tokens

        self.tokens = nn.Parameter(torch.empty(1, num_tokens, channels))
        self.mlp_token2feat = nn.Linear(channels, channels)
        self.mlp_delta_f = nn.Linear(channels, channels)

        val = math.sqrt(6.0 / float(3 * channels + channels))
        nn.init.uniform_(self.tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        
        self.scale = nn.Parameter(torch.tensor(scale_init))
        
        self.gate_gen = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.freq_k = freq_k
        self.out_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.dgf = DecoupledGatedFusion(channels)

    def forward_depthforge_delta(self, feats, depth_feats, tokens, fusion):
       
        attn_lf = torch.matmul(feats, tokens.transpose(1, 2))
        attn_hf = torch.matmul(depth_feats, tokens.transpose(1, 2))
        
        scale_factor = self.channels ** -0.5
        attn_lf = attn_lf * scale_factor
        attn_hf = attn_hf * scale_factor
        
        attn_total = attn_lf + attn_hf 
        attn_map = F.softmax(attn_total, dim=-1) 
        
        # 2. Aggregation
        tokens_trans = self.mlp_token2feat(tokens) 
        delta_f = torch.matmul(attn_map, tokens_trans)
        
        # 3. Projection
        delta_f = self.mlp_delta_f(delta_f)
        return delta_f

    def forward(self, x_hf, x_lf):
        
        B, C, D, H, W = x_hf.shape
        base_fused = self.dgf(x_lf=x_lf, x_hf=x_hf)
        
        flat_hf = x_hf.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        flat_lf = x_lf.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        tokens = self.tokens.expand(B, -1, -1)
        
        flat_delta = self.forward_depthforge_delta(flat_hf, flat_lf, tokens, base_fused)
        flat_delta = flat_delta * self.scale
        
        feat_delta = flat_delta.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
       
        # Gate selection
        gate = self.gate_gen(feat_delta)
        out = base_fused * gate + base_fused
        
        return out
