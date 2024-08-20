import torch
import torch.nn as nn
import torch.nn.functional as F

class FastDivergenceRotorLoss(nn.Module):
    def __init__(self, weight_div=1, weight_rot=1, dim_norm=1):
        super(FastDivergenceRotorLoss, self).__init__()
        self.weight_div = weight_div
        self.weight_rot = weight_rot
        self.dim_norm = dim_norm

    def divergence(self, vec1, vec2):
        # Упрощенная дивергенция: просто разница между суммами элементов
        return torch.abs(torch.sum(vec1 - vec2, dim=self.dim_norm))

    def rotor(self, vec1, vec2):
        # Упрощенный ротор: разница между произведениями соответствующих элементов
        return torch.abs(torch.sum(vec1 * vec2, dim=self.dim_norm))

    def forward(self, init_img_vec, next_img_vec, init_unclip, pred_unclip):
        device = pred_unclip.device
        diff_img = (init_img_vec - next_img_vec).squeeze(dim=1).to(device).to(torch.float32)
        diff_unclip = (init_unclip.to(device).to(torch.float32) - pred_unclip).squeeze(dim=1)

        diff_img_norm = F.normalize(diff_img, dim=self.dim_norm)
        diff_unclip_norm = F.normalize(diff_unclip, dim=self.dim_norm)

        # Вычисляем упрощенную дивергенцию и ротор для каждой пары
        div_loss = self.divergence(diff_img_norm.T, diff_unclip_norm.T)
        rot_loss = self.rotor(diff_img_norm.T, diff_unclip_norm.T)

        return self.weight_div * div_loss, self.weight_rot * rot_loss