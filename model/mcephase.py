import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.models as models

from utils import read_config_file
config_path = 'config.json'
config = read_config_file(config_path)

seq_len = config['data']['seq_len']
n_feature = 1024
output_size = 1  # output size = 1 to predict the mce phase score, ES=0 ED=1
enc_num_layer = 2
dec_num_layer = 1
enc_hidden = 512
dec_hidden = 512


class Attention(nn.Module):
    def __init__(self, enc_hidden, dec_hidden):
        """
           因为encoder是bidirectional的,所以他是2*hidden_dim
        """
        super(Attention, self).__init__()
        # hidden_dim*2: output_enc + hiddim_dim: hidden
        self.attn = nn.Linear(
            (enc_hidden * 2) + dec_hidden, dec_hidden, bias=False)
        self.v = nn.Linear(dec_hidden, 1, bias=False)

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


class Encoder(nn.Module):
    def __init__(self, n_features, enc_hidden, dec_hidden, enc_num_layer):
        super(Encoder, self).__init__()

        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.enc_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.hidden_layer = nn.Linear(enc_hidden*2, dec_hidden)
        self.cell_layer = nn.Linear(enc_hidden*2, dec_hidden)
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


class Decoder(nn.Module):
    def __init__(self, n_features, enc_hidden, dec_hidden, output_size, dec_num_layers):
        """
          the input size 
        """
        super(Decoder, self).__init__()
        self.dec_hidden = dec_hidden
        self.output_size = output_size
        self.dec_num_layers = dec_num_layers
        self.attention = Attention(enc_hidden, dec_hidden)
        self.lstm = nn.LSTM(n_features + dec_hidden*2,
                            dec_hidden, dec_num_layers, batch_first=True)
        self.fc = nn.Linear(dec_hidden, output_size)
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
    def __init__(self, n_features, enc_hidden, dec_hidden) -> None:
        super(Seq2Seq, self, ).__init__()
        self.encoder = Encoder(n_features, enc_hidden,
                               dec_hidden, enc_num_layer=enc_num_layer)
        self.decoder = Decoder(n_features, enc_hidden, dec_hidden,
                               output_size=output_size, dec_num_layers=dec_num_layer)

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
    def __init__(self, n_features=n_feature, enc_hidden=512, dec_hidden=512, output_dim=1):
        super(AttentionMcePhase, self).__init__()
        self.resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(
            self.resnet18.fc.in_features, n_features
        )
        self.bn1 = nn.BatchNorm1d(n_features)
        self.seq2seq = Seq2Seq(n_features=n_features,
                               enc_hidden=enc_hidden, dec_hidden=dec_hidden)

    def forward(self, x):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.resnet18(conv_input)
        x = self.bn1(conv_output)
        x = x.view(batch_size, timesteps, -1)
        batch_size, seq_len, _ = x.size()
        x = self.seq2seq(x)

        return x
