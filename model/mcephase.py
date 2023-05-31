import torch
import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F

seq_len = 30
class McePhase(nn.Module):
    def __init__(self, lstm_hidden_size=256, lstm_num_layers=8):
        super(McePhase, self).__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True )
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        # Extract features using ResNet18
        x = x.view(-1, 3, 256, 256)
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        # LSTM input shape: (batch_size, seq_length, input_size)
        x = x.view(-1, seq_len, 512)

        x = F.relu(x)
        # Feed sequence of features to LSTM
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        # Get final output from LSTM and pass through linear layer
        # fc_out = self.fc(final_out)
        x = self.fc(x).squeeze(dim=2)
        x = nn.functional.sigmoid(x) 
        return x


