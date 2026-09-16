import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialWeighted_DiceLoss(nn.Module):
    """
    Spatially Weighted Dice Loss
    Loss = 1 - Dice
    """
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred_probs, target, weight_map=None):
        """
        Args:
            pred_probs: [B, C, D, H, W]
            target:     [B, C, D, H, W] or [B, D, H, W]
            weight_map: [B, 1, D, H, W]
        """
        if target.dim() == 4:
            y_true = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        else:
            y_true = target

        y_pred = pred_probs

        if weight_map is None:
            weight_map = 1.0
        else:
            weight_map = weight_map.detach()

        intersection = torch.sum(weight_map * y_true * y_pred, dim=(2, 3, 4))
        y_true_sq = torch.sum(weight_map * y_true.pow(2), dim=(2, 3, 4))
        y_pred_sq = torch.sum(weight_map * y_pred.pow(2), dim=(2, 3, 4))

        dice = (2.0 * intersection + self.smooth) / (y_true_sq + y_pred_sq + self.smooth)

        return 1.0 - torch.mean(dice)


class DynamicLossConstraint(nn.Module):
    """
    Drift-aware Local Consistency (DLC)

    Confidence:
        C = 1 - (H - H_min) / (H_max - H_min + eps)

    Feature deviation:
        D_cos = 1 - cosine(f_clean, f_aug)

    Epoch-wise annealing:
        omega(t) = t / t_max

    Final weight:
        W_DLC = C^omega(t) * (1 + omega(t) * D_cos)
    """
    def __init__(self, num_classes, feat_channels=64, total_epochs=300):
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.total_epochs = total_epochs

    def compute_cosine_similarity(self, f1, f2):
        """Compute voxel-wise cosine similarity."""
        f1_norm = F.normalize(f1, dim=1)
        f2_norm = F.normalize(f2, dim=1)
        sim = torch.sum(f1_norm * f2_norm, dim=1, keepdim=True)
        return torch.clamp(sim, -1.0, 1.0)

    def compute_confidence(self, probs):
        """Compute normalized voxel-wise confidence."""
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)

        B = entropy.shape[0]
        ent_flat = entropy.view(B, -1)

        min_val = ent_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)
        max_val = ent_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)

        confidence = 1.0 - (entropy - min_val) / (max_val - min_val + 1e-8)

        return confidence

    def get_annealing_weight(self, feat_clean, feat_aug, pred_aug, current_epoch):
        """Compute the DLC weight."""
        sim = self.compute_cosine_similarity(feat_clean, feat_aug)
        drift = 1.0 - sim
        conf = self.compute_confidence(pred_aug)

        t_max = max(1, self.total_epochs - 1)
        progress = min(1.0, max(0.0, current_epoch / t_max))

        gate_term = torch.pow(conf, progress)
        drift_term = 1.0 + progress * drift

        weight = gate_term * drift_term

        return weight

    def forward(self,
                f_clean_src, f_aug_src, p_aug_src, mask_src,
                f_clean_tgt, f_aug_tgt, p_aug_tgt, mask_tgt,
                current_epoch=0):

        # ============================================================
        # 1. Spatial alignment
        # ============================================================
        if mask_src.dim() == 5:
            mask_src_idx = torch.argmax(mask_src, dim=1).long()
        else:
            mask_src_idx = mask_src.long()

        if mask_tgt.dim() == 5:
            mask_tgt_idx = torch.argmax(mask_tgt, dim=1).long()
        else:
            mask_tgt_idx = mask_tgt.long()

        target_shape_src = mask_src_idx.shape[-3:]
        target_shape_tgt = mask_tgt_idx.shape[-3:]

        if f_clean_src.shape[2:] != target_shape_src:
            f_clean_src = F.interpolate(f_clean_src, size=target_shape_src, mode='trilinear', align_corners=False)
            f_aug_src = F.interpolate(f_aug_src, size=target_shape_src, mode='trilinear', align_corners=False)

        if f_clean_tgt.shape[2:] != target_shape_tgt:
            f_clean_tgt = F.interpolate(f_clean_tgt, size=target_shape_tgt, mode='trilinear', align_corners=False)
            f_aug_tgt = F.interpolate(f_aug_tgt, size=target_shape_tgt, mode='trilinear', align_corners=False)

        # ============================================================
        # 2. Dynamic Loss Constraint
        # ============================================================
        w_src = self.get_annealing_weight(f_clean_src, f_aug_src, p_aug_src, current_epoch)
        w_tgt = self.get_annealing_weight(f_clean_tgt, f_aug_tgt, p_aug_tgt, current_epoch)

        # ============================================================
        # 3. Compatibility output
        # ============================================================
        loss_align = torch.tensor(0.0, device=f_clean_src.device)

        return w_src, w_tgt, loss_align
