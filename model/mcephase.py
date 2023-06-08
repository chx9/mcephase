import torch
import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .deeplab import AttentionDeeLabv3p

seq_len = 30
n_feature = 1024

class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.hidden_layer = nn.Linear(hidden_dim*2, hidden_dim)
        self.cell_layer = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x, (hidden, cell) = self.rnn1(x)

        x = self.dropout(x)
        # Concatenate the hidden states and cell states from both directions
        hidden = torch.cat(
            (hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]), 2)
        cell = torch.cat((cell[0:cell.size(0):2], cell[1:cell.size(0):2]), 2)

        # Reduce the dimension of the hidden states and cell states
        hidden = self.hidden_layer(hidden)
        cell = self.cell_layer(cell)

        return x, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        """
           因为encoder是bidirectional的,所以他是2*hidden_dim
        """
        super(Attention, self).__init__()
        # hidden_dim*2: output_enc + hiddim_dim: hidden
        self.attn = nn.Linear(hidden_dim*3, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, s, enc_output):
        """
            s: hidden state 
            enc_output: all output of encoder layer
        """
        b, seq_len, _ = enc_output.shape
        s = s.squeeze(dim=0)
        s = s.unsqueeze(1).repeat(1, seq_len, 1)

        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))
        atten = self.v(energy).squeeze(dim=2)
        soft_attn_weights = F.softmax(atten, dim=1)
        context = torch.bmm(soft_attn_weights.unsqueeze(
            dim=1), enc_output).squeeze(dim=1)
        return context, soft_attn_weights


class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dim, output_size, num_layers):
        """
          the input size 
        """
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.num_layers = num_layers
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(n_features + hidden_dim*2,
                            hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, enc_output, hidden, cell):
        context, _ = self.attention(hidden, enc_output)
        x = torch.cat((context, x), dim=1)
        x = x.unsqueeze(1)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.dropout(output)
        predictions = self.fc(output)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, n_features, hidden_dim) -> None:
        super(Seq2Seq, self, ).__init__()
        self.encoder = Encoder(n_features, hidden_dim)
        self.decoder = Decoder(n_features, hidden_dim,
                               output_size=1, num_layers=1)

    def forward(self, source):
        batch_size, seq_len, _ = source.shape
        outputs = torch.zeros(batch_size, seq_len, 1).to(
            source.device)  # output_size = 1
        enc_output, hidden, cell = self.encoder(source)
        for t in range(seq_len):
            x = source[:, t, :]
            output, hiden, cell = self.decoder(x, enc_output, hidden, cell)
            outputs[:, t] = output.squeeze(dim=1)
        return outputs.squeeze(dim=2)


class AttentionMcePhase(nn.Module):
    def __init__(self, n_features=n_feature, hidden_dim=512, output_dim=1):
        super(AttentionMcePhase, self).__init__()
        self.resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(
            self.resnet18.fc.in_features, n_features
        )
        self.bn1 = nn.BatchNorm1d(n_features)
        self.seq2seq = Seq2Seq(n_features=n_features, hidden_dim=hidden_dim)

    def forward(self, x):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.resnet18(conv_input)
        x = self.bn1(conv_output)
        x = x.view(batch_size, timesteps, -1)
        batch_size, seq_len, _ = x.size()
        x = self.seq2seq(x)

        return x
