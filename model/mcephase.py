import torch
import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
seq_len = 30
class McePhase(nn.Module):
    def __init__(self, lstm_hidden_size=512, lstm_num_layers=8):
        super(McePhase, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        # conv
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, self.lstm_hidden_size)

        self.bn1 = nn.BatchNorm1d(self.lstm_hidden_size)

        self.lstm = nn.LSTM(input_size=self.lstm_hidden_size, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(self.lstm_hidden_size*2)
        self.fc = nn.Linear(lstm_hidden_size*2, 30)
    def forward(self, x):
        # Extract features using ResNet18
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.resnet18(conv_input)
        lstm_input = self.bn1(conv_output)

        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_output, _  = self.lstm(lstm_input)
        lstm_output = self.bn2(lstm_output.transpose(1, 2)).transpose(1, 2) 
        lstm_output  = lstm_output[:, -1, :]
        out_put = self.fc(lstm_output)
        out_put = F.sigmoid(out_put)
        return out_put.squeeze(dim=0)


