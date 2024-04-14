import torch
from torch import nn as nn
from torch.nn import functional as F
import mmcv

from mmdet.models.builder import LOSSES
from mmdet.models.losses import FocalLoss, weight_reduce_loss

from einops import rearrange


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class MaskFocalLoss(FocalLoss):
    def __init__(self,**kwargs):
        super(MaskFocalLoss, self).__init__(**kwargs)
    
    def forward(self, 
                pred, 
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if not self.use_sigmoid:
            raise NotImplementedError
        
        num_classes = pred.size(1)
        loss = 0
        for index in range(num_classes):
            loss += self.loss_weight * py_sigmoid_focal_loss(
                pred[:,index],
                target[:,index],
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        loss /= num_classes
        return loss * self.loss_weight


@LOSSES.register_module()
class MaskDiceLoss(nn.Module):
    """Dice Loss PyTorch
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, loss_weight):
        super(MaskDiceLoss, self).__init__()
        self.smooth = 1e-5
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        bs, num_classes = pred.shape[:2]
        pred = rearrange(pred, 'b n h w -> b n (h w)')
        target = rearrange(target, 'b n h w -> b n (h w)')
        pred = pred.sigmoid()
        intersection = torch.sum(pred * target, dim=2)  # (N, C)
        union = torch.sum(pred.pow(2), dim=2) + torch.sum(target, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1
        
        loss = self.loss_weight * dice_loss
        return loss