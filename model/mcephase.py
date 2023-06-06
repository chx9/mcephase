import torch
import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .deeplab import AttentionDeeLabv3p
model_path = 'checkpoints/deeplab_attention.pth'
seq_len = 30


class McePhase(nn.Module):
    def __init__(self, lstm_hidden_size=2048, lstm_num_layers=8):
        super(McePhase, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        # conv
        self.resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(
            self.resnet18.fc.in_features, self.lstm_hidden_size)

        self.bn1 = nn.BatchNorm1d(self.lstm_hidden_size)

        self.lstm = nn.LSTM(input_size=self.lstm_hidden_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(self.lstm_hidden_size*2)
        self.fc = nn.Linear(lstm_hidden_size*2, 30)

    def forward(self, x):
        # Extract features using ResNet18
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.resnet18(conv_input)
        lstm_input = self.bn1(conv_output)

        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = self.bn2(lstm_output.transpose(1, 2)).transpose(1, 2)
        lstm_output = lstm_output[:, -1, :]
        out_put = self.fc(lstm_output)
        out_put = F.sigmoid(out_put)
        return out_put.squeeze(dim=0)



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # output shape: (batch_size, seq_len, hidden_size * 2)

        return out


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # output shape: (batch_size, seq_len, hidden_size)

        # Fully connected layer
        out = self.fc(out)  # output shape: (batch_size, seq_len, output_size)

        return out


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)

        return decoder_output


class AttentionMcePhase(nn.Module):
    def __init__(self, feature_num=1024):
        super(AttentionMcePhase, self).__init__()
        # conv
        self.resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(
            self.resnet18.fc.in_features, feature_num
        )
        self.bn1 = nn.BatchNorm1d(feature_num)
        self.seq2seq = Seq2Seq(input_size=feature_num, hidden_size=256)

    def forward(self, x):
        # Extract features using ResNet18
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.resnet18(conv_input)
        x = self.bn1(conv_output)
        x = x.view(batch_size, timesteps, -1)
        x = self.seq2seq(x)
        x = x.squeeze(dim=2)
        return x
