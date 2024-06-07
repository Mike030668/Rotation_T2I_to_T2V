import torch
import torch.nn as nn
import torch.nn.functional as F
from step_utils.rotations import RotationVectors

class TemporalConsistencyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(TemporalConsistencyLoss, self).__init__()
        self.eps = eps

    def forward(self, sequence):
        assert sequence.dim() == 3, f"Expected 3D tensor, got {sequence.dim()}D tensor instead"
        diff = sequence[1:, :, :] - sequence[:-1, :, :]
        diff = torch.clamp(diff, min=-1e6, max=1e6)
        loss = torch.mean(diff.pow(2))
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps
        return loss

class SequenceSmoothnessLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SequenceSmoothnessLoss, self).__init__()
        self.eps = eps

    def forward(self, sequence):
        assert sequence.dim() == 3, f"Expected 3D tensor, got {sequence.dim()}D tensor instead"
        diff1 = sequence[1:, :, :] - sequence[:-1, :, :]
        diff1 = torch.clamp(diff1, min=-1e6, max=1e6)
        diff2 = diff1[1:, :, :] - diff1[:-1, :, :]
        diff2 = torch.clamp(diff2, min=-1e6, max=1e6)
        loss = torch.mean(diff2.pow(2))
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps
        return loss

class CombinedLoss_base(nn.Module):
    def __init__(self, weight_rote=0.5, weight_mse=0.5, cos_way=-1, dim_norm=1, eps=1e-8):
        super(CombinedLoss_base, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.cos_loss = nn.CosineEmbeddingLoss(reduction='none')
        self.weight_mse = weight_mse
        self.weight_rote = weight_rote
        self.cos_way = cos_way
        self.dim_norm = dim_norm
        self.eps = eps

    def forward(self, init_img_vec, next_img_vec, init_unclip, pred_unclip):
        device = pred_unclip.device
        diff_img = (init_img_vec - next_img_vec).squeeze(dim=1).to(device).to(torch.float32)
        diff_unclip = (init_unclip.squeeze(dim=1).to(device).to(torch.float32) - pred_unclip.squeeze(dim=1))
        target = torch.ones(diff_img.shape[-1]).to(device)
        diff_img_norm = F.normalize(diff_img, dim=self.dim_norm)
        diff_unclip_norm = F.normalize(diff_unclip, dim=self.dim_norm)

        if self.cos_way == 1:
            cos_loss = 1 - self.cos_loss(diff_img_norm.T, diff_unclip_norm.T, target.to(device))
        elif self.cos_way == -1:
            cos_loss = self.cos_loss(diff_img_norm.T, diff_unclip_norm.T, (-1)*target.to(device))
        else:
            raise ValueError("cos_way must be 1 or -1")

        mse_loss = self.mse_loss(diff_img, diff_unclip)
        mse_loss = torch.mean(mse_loss, dim=0)

        cos_loss = torch.nan_to_num(cos_loss, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps
        mse_loss = torch.nan_to_num(mse_loss, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps

        return self.weight_rote * cos_loss, self.weight_mse * mse_loss

class CombinedLossBaseWithTemporal(nn.Module):
    def __init__(self, base_loss_fn, weight_rote=0.5, weight_mse=0.5, weight_temporal=0.5, weight_smoothness=0.5, eps=1e-8):
        super(CombinedLossBaseWithTemporal, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.temporal_consistency_loss = TemporalConsistencyLoss(eps=eps)
        self.sequence_smoothness_loss = SequenceSmoothnessLoss(eps=eps)
        self.weight_rote = weight_rote
        self.weight_mse = weight_mse
        self.weight_temporal = weight_temporal
        self.weight_smoothness = weight_smoothness
        self.eps = eps

    def forward(self, init_img_vec, next_img_vec, init_unclip, pred_unclip):
        base_cos_loss, base_mse_loss = self.base_loss_fn(init_img_vec, next_img_vec, init_unclip, pred_unclip)

        pred_unclip_3d = pred_unclip.unsqueeze(1) if pred_unclip.dim() == 2 else pred_unclip

        temporal_loss = self.temporal_consistency_loss(pred_unclip_3d)
        smoothness_loss = self.sequence_smoothness_loss(pred_unclip_3d)

        combined_loss_cos = base_cos_loss + self.weight_temporal * temporal_loss + self.weight_smoothness * smoothness_loss
        combined_loss_mse = base_mse_loss + self.weight_temporal * temporal_loss + self.weight_smoothness * smoothness_loss

        combined_loss_cos = torch.nan_to_num(combined_loss_cos, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps
        combined_loss_mse = torch.nan_to_num(combined_loss_mse, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps

        return combined_loss_cos, combined_loss_mse



##############################################################################################################

############### Neuman_losses ##############################   

class MEGUpdate(nn.Module):
    def __init__(self, eta=0.1):
        super(MEGUpdate, self).__init__()
        self.eta = eta

    def forward(self, R, grad):
        skew_grad = grad - grad.transpose(-1, -2)
        log_R = torch.log(R)
        update = log_R - self.eta * skew_grad
        new_R = torch.matrix_exp(update)
        return new_R


class CombinedLossBaseMEG(nn.Module):
    def __init__(self, weight_rote=0.5, weight_mse=0.5, eta=0.01, cos_way=-1, dim_norm=1, eps=1e-8):
        super(CombinedLossBaseMEG, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.cos_loss = nn.CosineEmbeddingLoss(reduction='none')
        self.weight_mse = weight_mse
        self.weight_rote = weight_rote
        self.eta = eta
        self.cos_way = cos_way
        self.dim_norm = dim_norm
        self.eps = eps

    def meg_update(self, R, grad):
        # Assuming grad is of shape (batch_size, feature_dim)
        batch_size, feature_dim = grad.shape

        # Ensure R is of correct shape (feature_dim, feature_dim)
        if R.shape[0] != feature_dim or R.shape[1] != feature_dim:
            R = torch.eye(feature_dim, device=grad.device)

        # MEG update
        skew_grad = grad.unsqueeze(2) - grad.unsqueeze(1)  # Create a skew-symmetric matrix from the grad
        skew_grad = (skew_grad - skew_grad.transpose(-1, -2)) / 2.0  # Ensure it is skew-symmetric
        log_R = torch.log(R)
        update = log_R - self.eta * skew_grad.mean(dim=0)
        new_R = torch.matrix_exp(update)  # Use matrix exponential for SO(n) update
        return new_R

    def forward(self, init_img_vec, next_img_vec, init_unclip, pred_unclip):
        device = pred_unclip.device
        diff_img = (init_img_vec - next_img_vec).squeeze(dim=1).to(device).to(torch.float32)
        diff_img.requires_grad_()
        diff_unclip = (init_unclip.squeeze(dim=1).to(device).to(torch.float32) - pred_unclip.squeeze(dim=1))
        diff_unclip.requires_grad_()
        target = torch.ones(diff_img.shape[0]).to(device)
        diff_img_norm = F.normalize(diff_img, dim=self.dim_norm)
        diff_unclip_norm = F.normalize(diff_unclip, dim=self.dim_norm)

        if self.cos_way == 1:
            cos_loss = 1 - self.cos_loss(diff_img_norm, diff_unclip_norm, target.to(device))
        elif self.cos_way == -1:
            cos_loss = self.cos_loss(diff_img_norm, diff_unclip_norm, (-1) * target.to(device))
        else:
            raise ValueError("cos_way must be 1 or -1")

        mse_loss = self.mse_loss(diff_img, diff_unclip)
        mse_loss = torch.mean(mse_loss, dim=0)

        cos_loss = torch.nan_to_num(cos_loss, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps
        mse_loss = torch.nan_to_num(mse_loss, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps

        # Convert losses to scalar before computing gradients
        cos_loss_scalar = torch.mean(cos_loss)
        mse_loss_scalar = torch.mean(mse_loss)

        # Update rotation matrices with MEG
        R = torch.eye(diff_img.shape[-1], device=device)  # Initial rotation matrix
        grad = torch.autograd.grad(cos_loss_scalar, diff_img, retain_graph=True)[0]
        R = self.meg_update(R, grad)

        # Apply the rotation matrix to diff_img
        rotated_diff_img = torch.matmul(diff_img, R)

        # Recalculate the losses with rotated diff_img
        rotated_diff_img_norm = F.normalize(rotated_diff_img, dim=self.dim_norm)
        rotated_cos_loss = self.cos_loss(rotated_diff_img_norm, diff_unclip_norm, target.to(device))

        rotated_mse_loss = self.mse_loss(rotated_diff_img, diff_unclip)
        rotated_mse_loss = torch.mean(rotated_mse_loss, dim=0)

        rotated_cos_loss = torch.nan_to_num(rotated_cos_loss, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps
        rotated_mse_loss = torch.nan_to_num(rotated_mse_loss, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps

        return self.weight_rote * torch.mean(rotated_cos_loss), self.weight_mse * torch.mean(rotated_mse_loss)


class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()

    def forward(self, generated_sequence):
        # Calculate the difference between consecutive frames
        diff = generated_sequence[1:, :, :] - generated_sequence[:-1, :, :]
        # Compute the L2 norm of the differences
        loss = torch.mean(diff.pow(2))
        return loss

class SequenceSmoothnessLoss(nn.Module):
    def __init__(self):
        super(SequenceSmoothnessLoss, self).__init__()

    def forward(self, generated_sequence):
        # Calculate second-order differences (acceleration)
        diff1 = generated_sequence[1:, :, :] - generated_sequence[:-1, :, :]
        diff2 = diff1[1:, :, :] - diff1[:-1, :, :]
        # Compute the L2 norm of the second-order differences
        loss = torch.mean(diff2.pow(2))
        return loss


class CombinedLossBaseWithTemporalMEG(nn.Module):
    def __init__(self, base_loss_fn, weight_rote=0.5, weight_mse=0.5, weight_temporal=0.5, weight_smoothness=0.5, eps=1e-8):
        super(CombinedLossBaseWithTemporalMEG, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.temporal_consistency_loss = TemporalConsistencyLoss()
        self.sequence_smoothness_loss = SequenceSmoothnessLoss()
        self.weight_rote = weight_rote
        self.weight_mse = weight_mse
        self.weight_temporal = weight_temporal
        self.weight_smoothness = weight_smoothness
        self.eps = eps

    def forward(self, init_img_vec, next_img_vec, init_unclip, pred_unclip):
        base_cos_loss, base_mse_loss = self.base_loss_fn(init_img_vec, next_img_vec, init_unclip, pred_unclip)

        pred_unclip_3d = pred_unclip.unsqueeze(1) if pred_unclip.dim() == 2 else pred_unclip

        temporal_loss = self.temporal_consistency_loss(pred_unclip_3d)
        smoothness_loss = self.sequence_smoothness_loss(pred_unclip_3d)

        combined_loss_cos = base_cos_loss + self.weight_temporal * temporal_loss + self.weight_smoothness * smoothness_loss
        combined_loss_mse = base_mse_loss + self.weight_temporal * temporal_loss + self.weight_smoothness * smoothness_loss

        combined_loss_cos = torch.nan_to_num(combined_loss_cos, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps
        combined_loss_mse = torch.nan_to_num(combined_loss_mse, nan=0.0, posinf=1e6, neginf=-1e6) + self.eps

        # Convert combined losses to scalar before returning
        combined_loss_cos_scalar = torch.mean(combined_loss_cos)
        combined_loss_mse_scalar = torch.mean(combined_loss_mse)

        return combined_loss_cos_scalar, combined_loss_mse_scalar




##############################################################################################################

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
        one_target = torch.ones(vec_1.shape).to(vec_1.device)
        I = torch.eye(vec_1.shape[-1]).unsqueeze(0).repeat(vec_1.shape[0],1,1).to(vec_1.device)


        # Apply the transformation to target
        u_transform_a = torch.bmm(transform_a, u_vec.unsqueeze(-1)).squeeze(-1)
        u_transform_b = torch.bmm(transform_b, u_vec.unsqueeze(-1)).squeeze(-1)
        u_transform_I = torch.bmm(I, u_vec.unsqueeze(-1)).squeeze(-1)


        # Calculate MSE loss
        loss_a = self.alpha*self.mse_loss(u_transform_a, zer_target)
        loss_b = self.betta*self.mse_loss(u_transform_b, zer_target)
        
        loss = self.mse_loss(u_transform_I, one_target) + loss_a + loss_b #

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
    

#########################################################
class TransformLoss(nn.Module):
    def __init__(self, weight_rote=0.5, weight_mse=0.5, alpha = 1, betta = 1):
        super(TransformLoss, self).__init__()
        self.rotation_vectors = RotationVectors()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.alpha = alpha
        self.betta = betta
        self.weight_rote = weight_rote
        self.weight_mse = weight_mse

    def forward(self,  init_img_vec, next_img_vec, init_unclip, pred_unclip):

        device = pred_unclip.device
        diff_img =  (init_img_vec - next_img_vec).squeeze(dim=1).to(device).to(torch.float32) #
        diff_unclip = (init_unclip.squeeze(dim=1).to(device).to(torch.float32) - pred_unclip.squeeze(dim=1))

        mask = torch.norm(diff_img, dim=-1, keepdim=True) > 1e-8
        vec_1 = diff_img[mask.squeeze(1)]
        vec_2 = diff_unclip[mask.squeeze(1)]
        


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
        transform = self.alpha*transform_a + self.betta*transform_b
        u_vec = torch.ones(vec_1.shape).to(device)
        zer_target = torch.zeros(vec_1.shape).to(device)
        one_target = torch.ones(vec_1.shape).to(device)
        I = torch.eye(vec_1.shape[-1]).unsqueeze(0).repeat(vec_1.shape[0],1,1).to(device)


        # Apply the transformation to target
        u_transform = torch.bmm(transform, u_vec.unsqueeze(-1)).squeeze(-1)
        #u_transform_b = torch.bmm(transform_b, u_vec.unsqueeze(-1)).squeeze(-1)
        u_transform_I = torch.bmm(I, u_vec.unsqueeze(-1)).squeeze(-1)


        # Calculate MSE loss
        loss_u = torch.mean(self.mse_loss(u_transform, zer_target), dim=0)
        loss_I = torch.mean(self.mse_loss(u_transform_I, one_target), dim=0)
    
        loss_u = torch.nan_to_num(loss_u, nan=0.0, posinf=0.1, neginf = -0.1)
        loss_I = torch.nan_to_num(loss_I, nan=0.0, posinf=0.1, neginf = -0.1)

        # Calculate MSE loss
        mse_loss = self.mse_loss(diff_img, diff_unclip)  # Shape (None, 1, 1280)
        mse_loss = torch.mean(mse_loss, dim=0) + loss_I # Reduce to Shape (None, 1) but keep last dim for matching

        return self.weight_rote * loss_u, self.weight_mse * mse_loss



#######################################

class TransRoteLoss(nn.Module):
    def __init__(self, alpha = 1., betta = 1.):
        super(TransRoteLoss, self).__init__()
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
        one_target = torch.ones(vec_1.shape).to(vec_1.device)
        I = torch.eye(vec_1.shape[-1]).unsqueeze(0).repeat(vec_1.shape[0],1,1).to(vec_1.device)


        # Apply the transformation to target
        u_transform_a = torch.bmm(transform_a, u_vec.unsqueeze(-1)).squeeze(-1)
        u_transform_b = torch.bmm(transform_b, u_vec.unsqueeze(-1)).squeeze(-1)
        u_transform_I = torch.bmm(I, u_vec.unsqueeze(-1)).squeeze(-1)


        # Calculate MSE loss
        loss_a = self.alpha*self.mse_loss(u_transform_a, zer_target)
        loss_b = self.betta*self.mse_loss(u_transform_b, zer_target)
        loss_RT =  torch.mean(loss_a + loss_b, dim=0)
        loss_RT = torch.nan_to_num(loss_RT, nan=0.0, posinf=0.1, neginf = -0.1)

        
        loss_I = self.mse_loss(u_transform_I, one_target)
        loss_I = torch.mean(loss_I, dim=0)
        loss_I = torch.nan_to_num(loss_I, nan=0.0, posinf=0.1, neginf = -0.1)

        return loss_I, loss_RT


class CombinedLoss_RT(nn.Module):
    def __init__(self, weight_rote=0.5, weight_mse =0.5, alpha = 1.,  betta = 1., cos_way = -1, dim_norm = -1):
        super(CombinedLoss_RT, self).__init__()

        self.mse_loss = nn.MSELoss(reduction='none')
        self.cos_loss = nn.CosineEmbeddingLoss(reduction='none')
        self.trans_loss = TransRoteLoss(alpha, betta)
        self.weight_mse = weight_mse
        self.weight_rote = weight_rote
        self.cos_way = cos_way
        self.dim_norm = dim_norm
        self.alpha = alpha 
        self.betta = betta
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



        if self.alpha or self.betta:
            I_loss, RT_loss = self.trans_loss(diff_img, diff_unclip)  # Expected shape (None, 1280)

        return self.weight_rote * (cos_loss + RT_loss), self.weight_mse * (mse_loss + I_loss)