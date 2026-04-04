import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialWeighted_DiceLoss(nn.Module):
    """
    [Ours] 纯净版: 支持空间加权的 Dice Loss
    移除了交叉熵，仅保留 Dice。
    逻辑: Loss = 1 - Dice
    Dice = (2 * sum(w * y_p * y_t) + smooth) / (sum(w * y_p^2) + sum(w * y_t^2) + smooth)
    """
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred_probs, target, weight_map=None):
        """
        Args:
            pred_probs: [B, C, D, H, W] (Softmax后的概率)
            target:     [B, C, D, H, W] (One-hot) OR [B, D, H, W] (Index)
            weight_map: [B, 1, D, H, W] (DDSA 生成的权重图)
        """
        # 1. 标签处理: 强制转为 One-Hot [B, C, D, H, W]
        if target.dim() == 4:
            y_true = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        else:
            y_true = target

        y_pred = pred_probs

        # 2. 权重处理
        if weight_map is None:
            weight_map = 1.0
        else:
            # 确保权重不回传梯度，只作为系数
            weight_map = weight_map.detach()
            # 广播机制会自动处理 [B, 1, D, H, W] * [B, C, D, H, W]

        # 3. 计算加权 Dice (Sample-wise, Square Sum)
        # 所有的 Sum 都在空间维度 (2, 3, 4) 进行，保留 Batch 和 Class 维度
        
        # 分子: 2 * sum(w * y_t * y_p)
        intersection = torch.sum(weight_map * y_true * y_pred, dim=(2, 3, 4))
        
        # 分母: sum(w * y_t^2) + sum(w * y_p^2)
        y_true_sq = torch.sum(weight_map * y_true.pow(2), dim=(2, 3, 4))
        y_pred_sq = torch.sum(weight_map * y_pred.pow(2), dim=(2, 3, 4))
        
        # 计算 Dice Score [B, C]
        # 如果 weight_map 接近 0 (被抑制的噪声样本)，
        # 分子分母都为 0，结果 = smooth / smooth = 1.0。
        # 此时 Loss = 1 - 1 = 0。实现"完全忽略"。
        dice = (2.0 * intersection + self.smooth) / (y_true_sq + y_pred_sq + self.smooth)
        
        # 4. 计算 Mean Dice Loss (1 - Dice)
        return 1.0 - torch.mean(dice)


class DynamicTemporalConstraint(nn.Module):
    """
    [Ours] 动态退火版 (Dynamic Annealing DDSA)
    
    核心机制：将训练轮数 (Epoch) 引入权重公式。
    公式: W = Conf^lambda * (1 + lambda * alpha * (1-Sim)^beta)
    其中 lambda 随 epoch 从 0 增长到 1。
    
    效果：
    - 前期 (lambda->0): W ≈ 1。权重机制休眠，模型全速学习基础特征。
    - 后期 (lambda->1): 权重机制全开。严厉惩罚低置信度样本，重奖高难漂移样本。
    """
    def __init__(self, num_classes, alpha=1.0, beta=2.0, feat_channels=64, tau=None, warmup_epochs=200):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha # 奖励幅度
        self.beta = beta   # 敏感度控制
        self.tau = tau     # 兼容参数(不使用)
        self.warmup_epochs = warmup_epochs # 预热/退火周期

    def compute_cosine_similarity(self, f1, f2):
        """计算全局余弦相似度 [B, 1, D, H, W]"""
        f1_norm = F.normalize(f1, dim=1)
        f2_norm = F.normalize(f2, dim=1)
        # 截断到 [0, 1] 防止负相似度影响公式
        sim = torch.sum(f1_norm * f2_norm, dim=1, keepdim=True)
        return torch.clamp(sim, 0, 1)

    def compute_confidence(self, probs):
        """计算归一化置信度 [0, 1]"""
        # 1. 计算熵 Entropy = - sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
        
        # 2. Min-Max 归一化 (Batch内或Sample内)
        B = entropy.shape[0]
        ent_flat = entropy.view(B, -1)
        
        min_val = ent_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)
        max_val = ent_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)
        
        ent_norm = (entropy - min_val) / (max_val - min_val + 1e-8)
        
        # 3. 置信度 F = 1 - Norm_Entropy
        return 1.0 - ent_norm

    def get_annealing_weight(self, feat_clean, feat_aug, pred_aug, current_epoch):
        """
        核心动态退火权重逻辑
        """
        # 1. 计算基础指标
        sim = self.compute_cosine_similarity(feat_clean, feat_aug)
        conf = self.compute_confidence(pred_aug)
        drift = 1.0 - sim
        
        # 2. 计算时间因子 lambda (0 -> 1)
        # 限制在 [0, 1] 之间，线性增长
        progress = min(1.0, max(0.0, current_epoch / self.warmup_epochs))
        
        # 3. 动态公式
        # W = Conf^progress * (1 + progress * alpha * drift^beta)
        
        # Part A: 门控项 (Gate)
        # progress=0 -> Conf^0 = 1 (不抑制)
        # progress=1 -> Conf^1 = Conf (全额抑制)
        gate_term = torch.pow(conf, progress)
        
        # Part B: 奖励项 (Reward)
        # progress=0 -> 1 + 0 = 1 (不奖励)
        # progress=1 -> 1 + alpha*drift^beta (全额奖励)
        reward_term = 1.0 + progress * self.alpha * torch.pow(drift, self.beta)
        
        weight = gate_term * reward_term
        
        return weight

    def forward(self, 
                f_clean_src, f_aug_src, p_aug_src, mask_src,
                f_clean_tgt, f_aug_tgt, p_aug_tgt, mask_tgt,
                current_epoch=0): # <--- [必须] 传入 current_epoch
        
        # ============================================================
        # 1. 维度修正与对齐 (解决 ValueError 的关键)
        # ============================================================
        
        # [处理 Mask]: 统一转为 Index 类型以便获取尺寸
        if mask_src.dim() == 5:
            mask_src_idx = torch.argmax(mask_src, dim=1).long()
        else:
            mask_src_idx = mask_src.long()
            
        if mask_tgt.dim() == 5:
            mask_tgt_idx = torch.argmax(mask_tgt, dim=1).long()
        else:
            mask_tgt_idx = mask_tgt.long()

        # [获取目标尺寸]: 强制只取最后 3 个维度 (D, H, W)
        target_shape_src = mask_src_idx.shape[-3:]
        target_shape_tgt = mask_tgt_idx.shape[-3:]

        # [插值]: 确保特征图和 Mask 尺寸一致
        if f_clean_src.shape[2:] != target_shape_src:
             f_clean_src = F.interpolate(f_clean_src, size=target_shape_src, mode='trilinear', align_corners=False)
             f_aug_src = F.interpolate(f_aug_src, size=target_shape_src, mode='trilinear', align_corners=False)
             
        if f_clean_tgt.shape[2:] != target_shape_tgt:
             f_clean_tgt = F.interpolate(f_clean_tgt, size=target_shape_tgt, mode='trilinear', align_corners=False)
             f_aug_tgt = F.interpolate(f_aug_tgt, size=target_shape_tgt, mode='trilinear', align_corners=False)

        # ============================================================
        # 2. 计算动态退火权重
        # ============================================================
        w_src = self.get_annealing_weight(f_clean_src, f_aug_src, p_aug_src, current_epoch)
        w_tgt = self.get_annealing_weight(f_clean_tgt, f_aug_tgt, p_aug_tgt, current_epoch)

        # ============================================================
        # 3. 对比损失 (已废弃)
        # ============================================================
        # 返回 0，保持 train.py 接口兼容
        loss_align = torch.tensor(0.0, device=f_clean_src.device)
            
        return w_src, w_tgt, loss_align