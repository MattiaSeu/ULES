from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F 
import cv2 as cv
import scipy.ndimage as spni
import numpy as np
import random 

""" CrossEntropy loss, usually used for semantic segmentation.
"""

class CrossEntropyLoss(nn.Module):

  def __init__(self, weights: Optional[List] = None):
    super().__init__()
    if weights is not None:
      weights = torch.Tensor(weights)
    self.criterion = nn.CrossEntropyLoss(weights)

  def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ Compute cross entropy loss.

    Args:
        inputs (torch.Tensor): unnormalized input tensor of shape [B x C x H x W]
        target (torch.Tensor): ground-truth target tensor of shape [B x H x W]

    Returns:
          torch.Tensor: weighted mean of the output losses.
    """

    loss = self.criterion(inputs, target)

    return loss

""" Generalized IoU loss.
"""

class mIoULoss(nn.Module):
  """ Define mean IoU loss.

  Props go to https://github.com/PRBonn/bonnetal/blob/master/train/tasks/segmentation/modules/custom_losses.py
  """
  def __init__(self, weight: List[float]):
    super().__init__()
    self.weight = nn.Parameter(torch.Tensor(weight), requires_grad=False)

  def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ Compute loss based on predictions/inputs and ground-truths/targets.

    Args:
        logits (torch.Tensor): Predictions of shape [N x n_classes x H x W]
        target (torch.Tensor): Ground-truths of shape [N x H x W]

    Returns:
        torch.Tensor: mean IoU loss
    """
    # get number of classes
    n_classes = int(logits.shape[1])

    # target to onehot
    target_one_hot = self.to_one_hot(target, n_classes) # [N x H x W x n_classes]

    batch_size = target_one_hot.shape[0]

    # map to (0,1)
    probs = F.softmax(logits, dim=1)
    
    # Numerator Product
    # inter = probs * target_one_hot * masking.unsqueeze(1)
    inter = probs * target_one_hot

    # Average over all pixels N x C x H x W => N x C
    inter = inter.view(batch_size, n_classes, -1).mean(2) + 1e-8

    # Denominator
    union = probs + target_one_hot - (probs * target_one_hot) + 1e-8
    # Average over all pixels N x C x H x W => N x C
    union = union.view(batch_size, n_classes, -1).mean(2)

    # Weights for loss
    frequency = target_one_hot.view(batch_size, n_classes, -1).sum(2).float()
    present = (frequency > 0).float()

    # -log(iou) is a good surrogate for loss
    loss = -torch.log(inter / union) * present * self.weight
    loss = loss.sum(1) / present.sum(1)  # pseudo average

    # Return average loss over batch
    return loss.mean()

  def to_one_hot(self, tensor: torch.Tensor, n_classes: int) -> torch.Tensor:
    """ Convert tensor to its one hot encoded version.

    Props go to https://github.com/PRBonn/bonnetal/blob/master/train/common/onehot.py

    Args:
      tensor (torch.Tensor): ground truth tensor of shape [N x H x W]
      n_classes (int): number of classes

    Returns:
      torch.Tensor: one hot tensor of shape [N x n_classes x H x W]
    """
    if len(tensor.size()) == 1:
        b = tensor.size(0)
        if tensor.is_cuda:
            one_hot = torch.zeros(b, n_classes, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(b, n_classes).scatter_(1, tensor.unsqueeze(1), 1)
    elif len(tensor.size()) == 2:
        n, b = tensor.size()
        if tensor.is_cuda:
            one_hot = torch.zeros(n, n_classes, b, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(n, n_classes, b).scatter_(1, tensor.unsqueeze(1), 1)
    elif len(tensor.size()) == 3:
        n, h, w = tensor.size()
        if tensor.is_cuda:
            one_hot = torch.zeros(n, n_classes, h, w, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.unsqueeze(1), 1)
    return one_hot
                                                                                                                          
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=.01, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, out, target):
        out = out.squeeze()
        target = target.squeeze()
        log_positives = (out[target != 0]).clamp(min=1e-4)
        log_negatives = (1 - out[target == 0]).clamp(min=1e-4)
        positives = -self.alpha * (1 - out[target != 0]) ** self.gamma * torch.log(log_positives)
        negatives = -(1 - self.alpha) * out[target == 0] ** self.gamma * torch.log(log_negatives)
        if len(positives) > 0 and len(negatives) > 0:
            return torch.mean(positives) + torch.mean(negatives)
        elif len(positives) > 0:
            return torch.mean(positives)
        return torch.mean(negatives)

def masks_to_centers(masks_original: torch.Tensor) -> torch.Tensor:
    if masks_original.numel() == 0:
        return torch.zeros((0, 4), device=masks_original.device, dtype=torch.float)

    tmp_masks = F.one_hot(masks_original.long()).permute(0,3,1,2)
    masks = tmp_masks[:,1:,:,:]
    B, num, H, W = masks.shape

    center_mask = torch.zeros( (B, H, W) , device=masks.device, dtype=torch.float)
    
    for batch_idx, mask in enumerate(masks):
        for submask in mask:
            if submask.sum() == 0:
                continue
            x, y = torch.where(submask != 0)
            xy = torch.cat([x.unsqueeze(0),y.unsqueeze(0)], dim=0)
            mu, _ = torch.median(xy,dim=1, keepdim=True)
            center_idx = torch.argmin(torch.sum(torch.abs(xy - mu), dim=0))
            center = xy[:,center_idx]
            center_mask[batch_idx, center[0], center[1]] = 1.
    return center_mask


import torch
import torch.nn as nn


# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5:
        return [2, 3, 4]

    # Two dimensional
    elif len(shape) == 4:
        return [2, 3]

    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


class SymmetricFocalLoss(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.7, gamma=2., epsilon=1e-07):
        super(SymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # Calculate losses separately for each class
        back_ce = torch.pow(1 - y_pred[:, 0, :, :], self.gamma) * cross_entropy[:, 0, :, :]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = torch.pow(1 - y_pred[:, 1, :, :], self.gamma) * cross_entropy[:, 1, :, :]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss


class AsymmetricFocalLoss(nn.Module):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.25
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.7, gamma=2., epsilon=1e-07):
        super(AsymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # Calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - y_pred[:, 0, :, :], self.gamma) * cross_entropy[:, 0, :, :]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:, 1, :, :]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss


class SymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1 - y_pred), axis=axis)
        fp = torch.sum((1 - y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1 - dice_class[:, 0]) * torch.pow(1 - dice_class[:, 0], -self.gamma)
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -self.gamma)

        # Average class scores
        loss = torch.mean(torch.stack([back_dice, fore_dice], axis=-1))
        return loss


class AsymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1 - y_pred), axis=axis)
        fp = torch.sum((1 - y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        # Calculate losses separately for each class, only enhancing foreground class
        back_dice = (1 - dice_class[:, 0])
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -self.gamma)

        # Average class scores
        loss = torch.mean(torch.stack([back_dice, fore_dice], axis=-1))
        return loss


class SymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super(SymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        symmetric_ftl = SymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)
        symmetric_fl = SymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)
        if self.weight is not None:
            return (self.weight * symmetric_ftl) + ((1 - self.weight) * symmetric_fl)
        else:
            return symmetric_ftl + symmetric_fl


class AsymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, weight=0.5, delta=0.6, gamma=0.2):
        super(AsymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # Obtain Asymmetric Focal Tversky loss
        asymmetric_ftl = AsymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)

        # Obtain Asymmetric Focal loss
        asymmetric_fl = AsymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)

        # Return weighted sum of Asymmetrical Focal loss and Asymmetric Focal Tversky loss
        if self.weight is not None:
            return (self.weight * asymmetric_ftl) + ((1 - self.weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl
