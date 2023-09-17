import torch
import torch.nn as nn

from utils import read_config_file
config_path = 'config.json'
config = read_config_file(config_path)

alpha = config['model_params']['alpha']
extrema_weight = config['model_params']['extrema_weight']
class CombineLoss(nn.Module):
    def __init__(self, alpha=alpha, extrema_weight=extrema_weight):
        super(CombineLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.extrema_weight = extrema_weight

    def mono_loss(self, preds, labels):
        diff_label = labels[:, 1:] - labels[:, :-1]
        diff_pred = preds[:, 1:] - preds[:, :-1]

        increase = (diff_label > 0).float()
        decrease = (diff_label < 0).float()

        loss_increase = increase * torch.clamp(-diff_pred, min=0)
        loss_decrease = decrease * torch.clamp(diff_pred, min=0)

        structure_loss = torch.mean(loss_increase + loss_decrease)
        return structure_loss

    def es_loss(self, preds, labels):
        es_mask = (labels == -0.2).float()
        es_loss = self.mse_loss(
            preds * es_mask, labels * es_mask) * self.extrema_weight
        return es_loss

    def ed_loss(self, preds, labels):
        ed_mask = (labels == 1.2).float()
        ed_loss = self.mse_loss(
            preds * ed_mask, labels * ed_mask) * self.extrema_weight
        return ed_loss

    def forward(self, preds, labels):
        mse_loss = self.mse_loss(preds, labels)
        mono_loss = self.mono_loss(preds, labels)
        structure_loss = (1-self.alpha) * mse_loss + self.alpha * mono_loss
        es_loss = self.es_loss(preds, labels)
        ed_loss = self.ed_loss(preds, labels)
        return structure_loss + es_loss + ed_loss
