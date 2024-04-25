import torch
import torch.nn as nn

class CosAccuracy(nn.Module):
  def __init__(self):
    super(CosAccuracy, self).__init__()
    self.cos_loss = nn.CosineEmbeddingLoss(reduction='mean')

  def forward(self, init_img_vec, next_img_vec, init_unclip, pred_unclip):

        # Assume vec_1, vec_2 are of shape (None, 1, 1280)
        init_img_vec = F.normalize(init_img_vec, dim = -1).squeeze(1).to(torch.float32)
        next_img_vec = F.normalize(next_img_vec, dim = -1).squeeze(1).to(torch.float32)
        init_unclip = F.normalize(init_unclip, dim = -1).squeeze(1).to(torch.float32)
        pred_unclip = F.normalize(pred_unclip, dim = -1).squeeze(1)

        device = pred_unclip.device
        target = torch.ones(init_img_vec.shape[-1]).to(device)

        cos_init = self.cos_loss(init_img_vec.T.to(device),
                                 init_unclip.T.to(device), target)

        # CosineEmbeddingLoss between difference
        cos_pred = self.cos_loss(next_img_vec.T.to(device),
                                 pred_unclip.T, target)

        return 2*cos_init*cos_pred/(cos_init + cos_pred + 1e-8)