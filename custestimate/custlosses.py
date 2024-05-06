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
    

############### CombinedLoss_trans ##############################

class TransformationBasedRotationLoss(nn.Module):
    def __init__(self, alpha = 0.7, betta = 0.3):
        super(TransformationBasedRotationLoss, self).__init__()
        self.rotation_vectors = RotationVectors()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.alpha = alpha
        self.betta = betta

    def forward(self, vec_1, vec_2):

        mask = torch.norm(vec_1, dim=-1, keepdim=True) > 1e-8
        vec_1 = vec_1[mask.squeeze(1)]
        vec_2 = vec_2[mask.squeeze(1)]


        # Calculate angle and unit vectors
        a = self.rotation_vectors.angle(vec_1, vec_2)
        n1 = self.rotation_vectors.unit_vector(vec_1)
        n2 = self.rotation_vectors.unit_vector(vec_2 - (n1 * (n1 * vec_2).sum(dim=1, keepdim=True)))

        # Calculate rotation transformation excluding the identity matrix component
        sin_a = torch.sin(a).unsqueeze(-1).unsqueeze(-1)
        cos_a_minus_1 = (torch.cos(a) - 1).unsqueeze(-1).unsqueeze(-1)
        n2n1T = n2.unsqueeze(2) * n1.unsqueeze(1)
        n1n2T = n1.unsqueeze(2) * n2.unsqueeze(1)
        n1n1T = n1.unsqueeze(2) * n1.unsqueeze(1)
        n2n2T = n2.unsqueeze(2) * n2.unsqueeze(1)

        transform_a = (n2n1T - n1n2T) * sin_a
        transform_b = (n1n1T + n2n2T) * cos_a_minus_1
        u_vec = torch.ones(vec_1.shape).to(vec_1.device)
        zer_target = torch.zeros(vec_1.shape).to(vec_1.device)


        # Apply the transformation to target
        u_transform_a = torch.bmm(transform_a, u_vec.unsqueeze(-1)).squeeze(-1)
        u_transform_b = torch.bmm(transform_b, u_vec.unsqueeze(-1)).squeeze(-1)

        # Calculate MSE loss
        loss_a = self.alpha*self.mse_loss(u_transform_a, zer_target)
        loss_b = self.betta*self.mse_loss(u_transform_b, zer_target)
        loss = loss_a + loss_b

        loss = torch.mean(loss, dim=0)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.1, neginf = -0.1)

        return loss



class CombinedLoss_trans(nn.Module):
    def __init__(self, weight_rote=0.5, weight_mse=0.5, alpha = 0.7, betta = 0.3):
        super(CombinedLoss_trans, self).__init__()
        self.rotation_loss = TransformationBasedRotationLoss(alpha, betta)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.weight_rote = weight_rote
        self.weight_mse = weight_mse

    def forward(self, init_img_vec, next_img_vec, init_unclip, pred_unclip ): #vec, target):

        device = pred_unclip.device
        diff_img =  (init_img_vec - next_img_vec).squeeze(dim=1).to(device).to(torch.float32) #
        diff_unclip = (init_unclip.squeeze(dim=1).to(device).to(torch.float32) - pred_unclip.squeeze(dim=1))

        # Calculate MSE loss
        mse_loss = self.mse_loss(diff_img, diff_unclip)  # Shape (None, 1, 1280)
        mse_loss = torch.mean(mse_loss, dim=0)  # Reduce to Shape (None, 1) but keep last dim for matching

        # Calculate rotation transformation loss
        rotation_loss = self.rotation_loss(diff_img, diff_unclip)  # Expected shape (None, 1)

        return self.weight_rote * rotation_loss, self.weight_mse * mse_loss

############### CombinedLoss_trans ##############################


class CombinedLoss_cos_trans(nn.Module):
    def __init__(self, weight_rote=0.5,
                  weight_mse=0.5, 
                  cos_way = -1, 
                  dim_norm = -1, 
                  alpha = 0.7, 
                  betta = 0.3):
        
        super(CombinedLoss_cos_trans, self).__init__()
        self.trans_loss = TransformationBasedRotationLoss(alpha, betta)
        self.cos_loss = nn.CosineEmbeddingLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.weight_rote = weight_rote
        self.weight_mse = weight_mse
        self.cos_way = cos_way
        self.dim_norm = dim_norm
        self.alfa = alpha
        self.betta = betta

    def forward(self, init_img_vec, next_img_vec, init_unclip, pred_unclip ): #vec, target):

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

        # Calculate MSE loss
        mse_loss = self.mse_loss(diff_img, diff_unclip)  # Shape (None, 1, 1280)
        mse_loss = torch.mean(mse_loss, dim=0)  # Reduce to Shape (None, 1) but keep last dim for matching

        # Calculate rotation transformation loss
        rotation_loss = cos_loss
        if self.alfa or self.betta:
            rotation_loss = rotation_loss +  self.trans_loss(diff_img, diff_unclip)  # Expected shape (None, 1280)

        return self.weight_rote * rotation_loss, self.weight_mse * mse_loss
    



#######################################

class TransformationRotationLoss(nn.Module):
    def __init__(self, alpha = 0.7, betta = 0.3):
        super(TransformationRotationLoss, self).__init__()
        self.rotation_vectors = RotationVectors()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.alpha = alpha
        self.betta = betta

    def forward(self, vec_1, vec_2):

        mask = torch.norm(vec_1, dim=-1, keepdim=True) > 1e-8
        vec_1 = vec_1[mask.squeeze(1)]
        vec_2 = vec_2[mask.squeeze(1)]


        # Calculate angle and unit vectors
        a = self.rotation_vectors.angle(vec_1, vec_2)
        n1 = self.rotation_vectors.unit_vector(vec_1)
        n2 = self.rotation_vectors.unit_vector(vec_2 - (n1 * (n1 * vec_2).sum(dim=1, keepdim=True)))

        # Calculate rotation transformation excluding the identity matrix component
        sin_a = torch.sin(a).unsqueeze(-1).unsqueeze(-1)
        cos_a_minus_1 = (torch.cos(a) - 1).unsqueeze(-1).unsqueeze(-1)
        n2n1T = n2.unsqueeze(2) * n1.unsqueeze(1)
        n1n2T = n1.unsqueeze(2) * n2.unsqueeze(1)
        n1n1T = n1.unsqueeze(2) * n1.unsqueeze(1)
        n2n2T = n2.unsqueeze(2) * n2.unsqueeze(1)

        transform_a = (n2n1T - n1n2T) * sin_a
        transform_b = (n1n1T + n2n2T) * cos_a_minus_1
        u_vec = torch.ones(vec_1.shape).to(vec_1.device)
        zer_target = torch.zeros(vec_1.shape).to(vec_1.device)

        I = torch.eye(vec_1.shape).to(vec_1.device)


        # Apply the transformation to target
        u_transform_a = torch.bmm(transform_a, u_vec.unsqueeze(-1)).squeeze(-1)
        u_transform_b = torch.bmm(transform_b, u_vec.unsqueeze(-1)).squeeze(-1)

        u_transform_I = torch.bmm(I, u_vec.unsqueeze(-1)).squeeze(-1)


        # Calculate MSE loss
        loss_a = self.alpha*self.mse_loss(u_transform_a, zer_target)
        loss_b = self.betta*self.mse_loss(u_transform_b, zer_target)
        
        loss = self.mse_loss(u_transform_I, zer_target) + loss_a + loss_b

        loss = torch.mean(loss, dim=0)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.1, neginf = -0.1)

        return loss

class CombinedLoss_cos_trans_I(nn.Module):
    def __init__(self, weight_rote=0.5,
                  weight_mse=0.5, 
                  cos_way = -1, 
                  dim_norm = -1, 
                  alpha = 0.7, 
                  betta = 0.3):
        
        super(CombinedLoss_cos_trans_I, self).__init__()
        self.trans_loss = TransformationRotationLoss(alpha, betta)
        self.cos_loss = nn.CosineEmbeddingLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.weight_rote = weight_rote
        self.weight_mse = weight_mse
        self.cos_way = cos_way
        self.dim_norm = dim_norm
        self.alfa = alpha
        self.betta = betta

    def forward(self, init_img_vec, next_img_vec, init_unclip, pred_unclip ): #vec, target):

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

        # Calculate MSE loss
        mse_loss = self.mse_loss(diff_img, diff_unclip)  # Shape (None, 1, 1280)
        mse_loss = torch.mean(mse_loss, dim=0)  # Reduce to Shape (None, 1) but keep last dim for matching

        # Calculate rotation transformation loss
        rotation_loss = cos_loss
        if self.alfa or self.betta:
            rotation_loss = rotation_loss +  self.trans_loss(diff_img, diff_unclip)  # Expected shape (None, 1280)

        return self.weight_rote * rotation_loss, self.weight_mse * mse_loss
    