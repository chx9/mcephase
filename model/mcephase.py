import torch
import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F

seq_len = 30
class McePhase(nn.Module):
    def __init__(self, lstm_hidden_size=512, lstm_num_layers=8):
        super(McePhase, self).__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 512)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden_size*2, 30)

    def forward(self, x):
        # Extract features using ResNet18
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.resnet18(conv_input)
        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_output, _  = self.lstm(lstm_input)
        lstm_output  = lstm_output[:, -1, :]
        out_put = self.fc(lstm_output)
        out_put = F.sigmoid(out_put)
        return out_put.squeeze(dim=0)


