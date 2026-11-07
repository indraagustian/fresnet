import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Multiclass Dice Loss with smooth factor.
    """
    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.clamp(min=0), num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice_score)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class PixelContrastiveLoss(nn.Module):
    """
    Supervised Pixel-Level Contrastive Loss (PCL).
    Pulls representations of the same class together while repelling distinct classes in the feature manifold.
    """
    def __init__(self, temperature: float = 0.1, max_samples: int = 128):
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples

    def forward(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)
        masks = masks.reshape(-1)
        
        classes = [0, 1, 2]
        sampled_features = {}
        for cls in classes:
            idx = (masks == cls).nonzero(as_tuple=True)[0]
            if len(idx) > self.max_samples:
                perm = torch.randperm(len(idx), device=features.device)[:self.max_samples]
                idx = idx[perm]
            if len(idx) > 0:
                cls_features = features[idx]
                sampled_features[cls] = F.normalize(cls_features, p=2, dim=1)
            else:
                sampled_features[cls] = torch.empty((0, C), device=features.device)
                
        valid_classes = 0
        total_loss = torch.tensor(0.0, device=features.device)
        for cls in [1, 2]:  # Crop and Weed foreground classes
            anchors = sampled_features[cls]
            if anchors.size(0) < 2:
                continue
            sim_pos = torch.matmul(anchors, anchors.T) / self.temperature
            mask_self = torch.eye(anchors.size(0), dtype=torch.bool, device=features.device)
            sim_pos_exp = torch.exp(sim_pos).masked_fill(mask_self, 0.0)
            
            neg_pool = []
            for neg_cls in classes:
                if neg_cls != cls and sampled_features[neg_cls].size(0) > 0:
                    neg_pool.append(sampled_features[neg_cls])
            if len(neg_pool) == 0:
                continue
            negs = torch.cat(neg_pool, dim=0)
            sim_neg = torch.matmul(anchors, negs.T) / self.temperature
            sim_neg_exp = torch.exp(sim_neg)
            
            pos_sum = sim_pos_exp.sum(dim=1)
            denom = pos_sum + sim_neg_exp.sum(dim=1) + 1e-8
            loss_cls = -torch.log((pos_sum / denom) + 1e-8).mean()
            total_loss = total_loss + loss_cls
            valid_classes += 1
            
        if valid_classes > 0:
            return total_loss / valid_classes
        return total_loss

class FRESNetCompositeLoss(nn.Module):
    """
    Composite segmentation loss combining Dice, Focal, and Pixel Contrastive losses:
    L_total = 0.45 * L_Dice + 0.45 * L_Focal + 0.10 * L_PCL
    """
    def __init__(self, dice_weight: float = 0.45, focal_weight: float = 0.45, cl_weight: float = 0.10,
                 gamma: float = 2.0, temperature: float = 0.1, max_samples: int = 128):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=gamma)
        self.pcl_loss = PixelContrastiveLoss(temperature=temperature, max_samples=max_samples)
        self.w_dice = dice_weight
        self.w_focal = focal_weight
        self.w_cl = cl_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, features: torch.Tensor = None):
        l_dice = self.dice_loss(logits, targets)
        l_focal = self.focal_loss(logits, targets)
        if features is not None and self.w_cl > 0:
            l_pcl = self.pcl_loss(features, targets)
        else:
            l_pcl = torch.tensor(0.0, device=logits.device)
        total = self.w_dice * l_dice + self.w_focal * l_focal + self.w_cl * l_pcl
        return total, {
            'loss_total': total.item(),
            'loss_dice': l_dice.item(),
            'loss_focal': l_focal.item(),
            'loss_pcl': l_pcl.item()
        }
