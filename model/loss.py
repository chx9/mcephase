import torch
import torch.nn as nn
class CombineLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombineLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def mono_loss(self, preds, labels):
        diff_label = labels[:, 1:] - labels[:, :-1]
        diff_pred = preds[:, 1:] - preds[:, :-1]

        increase = (diff_label > 0).float()
        decrease = (diff_label < 0).float()

        loss_increase = increase * torch.clamp(-diff_pred, min=0)
        loss_decrease = decrease * torch.clamp(diff_pred, min=0)

        total_loss = torch.mean(loss_increase + loss_decrease)
        return total_loss

    def forward(self, preds, labels):
        mse_loss = self.mse_loss(preds, labels)
        mono_loss = self.mono_loss(preds, labels)
        total_loss = (1-self.alpha)* mse_loss + self.alpha * mono_loss
        return total_loss