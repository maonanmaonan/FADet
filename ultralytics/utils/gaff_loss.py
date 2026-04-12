import torch 
import torch.nn as nn
from .metrics import probiou

def get_sigma(input, eps=1e-7):
    _, wh, theta = input.split([2, 2, 1], -1)
    wh = wh.clamp(min=eps)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    R = torch.cat((Cos, -Sin, Sin, Cos), -1).view(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    sigma = (R @ S.square() @ R.transpose(1, 2)).reshape(-1, 2, 2)
    return sigma

def compute_gwd(pred, target, eps=1e-7, alpha=1.0, tau=1.0, norm=True):
    pred_xy = pred[..., :2]
    target_xy = target[..., :2]
    pred_sigma = get_sigma(pred, eps)
    target_sigma = get_sigma(target, eps)
    # m calculate
    xy_dist = (pred_xy - target_xy).square().sum(-1)
    whr_dist = pred_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_dist = whr_dist + target_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_tr = (pred_sigma @ target_sigma).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (pred_sigma.det() * target_sigma.det()).clamp(0).sqrt()
    whr_dist = whr_dist + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()
    )
    dist = (xy_dist + alpha * alpha * whr_dist).clamp(0).sqrt()
    if norm:
        scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(eps)
        dist = dist / scale
    # loss = 1 - 1 / (tau + torch.log1p(dist))
    loss = 1 / (tau + torch.log1p(dist))
    return loss

    
def prob_gwd(target, pred, ground_truth):
    loss1 = probiou(pred, target)
    loss2 = compute_gwd(pred, target)
    alpha = adjust_alpha_by_area(ground_truth)  # 动态调整 alpha
    loss = alpha*loss2+(1-alpha)*loss1
    return loss


def adjust_alpha_by_area(ground_truth, area_threshold=90):
    """
    根据目标的面积动态调整 alpha。

    :param obb_pred: 预测的旋转框 (格式: [center_x, center_y, width, height, angle])
    :param obb_true: 真实的旋转框 (格式: [center_x, center_y, width, height, angle])
    :param alpha_min: 小目标时的最小 alpha
    :param alpha_max: 大目标时的最大 alpha
    :param area_threshold: 作为小目标的面积阈值
    :return: 动态调整后的 alpha
    """
    # 计算每个框的面积 (宽 * 高)
    area_true =ground_truth[:, 2] * ground_truth[:, 3]
    area = area_true
    # 根据面积动态调整 alpha
    alpha = torch.sigmoid(0.01*(area-area_threshold))
    alpha = torch.clamp(alpha, min=0.3, max=0.7)
    return alpha

