import torch
import torch.nn as nn
import torch.nn.functional as F
from step_utils.rotations import RotationVectors

class CombinedLoss_base(nn.Module):
    def __init__(self, weight_rote=0.5, weight_mse =0.5, cos_way = -1, dim_norm = 1):
        super(CombinedLoss_base, self).__init__()

        self.mse_loss = nn.MSELoss(reduction='none')
        self.cos_loss = nn.CosineEmbeddingLoss(reduction='none')
        self.weight_mse = weight_mse
        self.weight_rote = weight_rote
        self.cos_way = cos_way
        self.dim_norm = dim_norm

    def forward(self, init_img_vec, next_img_vec, init_unclip, pred_unclip):

        device = pred_unclip.device
        diff_img =  (init_img_vec - next_img_vec).squeeze(dim=1).to(device).to(torch.float32) #
        diff_unclip = (init_unclip.squeeze(dim=1).to(device).to(torch.float32) - pred_unclip.squeeze(dim=1))

        # Calculate cos_loss,
        target = torch.ones(diff_img.shape[-1])
        diff_img_norm = F.normalize(diff_img, dim = self.dim_norm)
        diff_unclip_norm = F.normalize(diff_unclip, dim = self.dim_norm)

        if self.cos_way == 1:
            cos_loss = 1 - self.cos_loss(diff_img_norm.T, diff_unclip_norm.T, target.to(device))  # Shape (None, 1, 1280)
        elif self.cos_way == -1:
            cos_loss = self.cos_loss(diff_img_norm.T, diff_unclip_norm.T, (-1)*target.to(device))  # Shape (None, 1, 1280)
        else: print("cos_way must be 1 or -1")

        # Calculate MSE for each element and then average across dimension 1 to match MSE shape
        mse_loss = self.mse_loss(diff_img, diff_unclip)  # Shape (None, 1, 1280)
        mse_loss = torch.mean(mse_loss, dim=0)#  # Reduce to Shape (None, 1)

        return self.weight_rote * cos_loss, self.weight_mse * mse_loss
    

class SumLosses(nn.Module):
    def __init__(self, set_losses, set_weights = None):
        super(SumLosses, self).__init__()

        self.set_losses = set_losses   
        self.set_weights = [1. for _ in range(len(set_losses))] if not set_weights else set_weights

    def forward(self, init_img_vec, next_img_vec, init_unclip, pred_unclip):
        
        diff_loss = 0
        rote_loss = 0
        for i, loss in enumerate(self.set_losses):

           rote, diff = loss(init_img_vec, next_img_vec, init_unclip, pred_unclip)
           rote_loss +=self.set_weights[i]*rote
           diff_loss +=self.set_weights[i]*diff

        return rote_loss, diff_loss