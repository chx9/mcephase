import torch
import torch.nn as nn

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
        out, (hidden_state, cell_state) = self.lstm(x, (h0, c0))  # output shape: (batch_size, seq_len, hidden_size * 2)

        return out, hidden_state, cell_state


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x, hidden_state, cell_state):
        # Forward propagate LSTM
        out, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))  # output shape: (batch_size, seq_len, hidden_size)

        # Fully connected layer
        out = self.fc(out)  # output shape: (batch_size, seq_len, output_size)

        return out, hidden_state, cell_state


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size, hidden_size, output_size, num_layers)

    def forward(self, x):
        encoder_output, encoder_hidden_state, encoder_cell_state = self.encoder(x)

        # Prepare the initial input for the decoder
        decoder_input = torch.zeros(x.size(0), 1, x.size(2)).to(x.device)  # Start with a zero tensor

        decoder_output = []
        for _ in range(x.size(1)):
            decoder_input, decoder_hidden_state, decoder_cell_state = self.decoder(decoder_input, encoder_hidden_state, encoder_cell_state)
            decoder_output.append(decoder_input)
            decoder_input = decoder_input[:, -1:, :]  # Use the last predicted output as the next input

        decoder_output = torch.cat(decoder_output, dim=1)  # Combine decoder outputs
        return decoder_output


# Example usage
batch_size = 16
seq_len = 30
feature_num = 1024
output_size = 1
hidden_size = 256
num_layers = 2

# Create random input tensor
input_tensor = torch.randn(batch_size, seq_len, feature_num)

# Create the model
model = Seq2Seq(feature_num, hidden_size, output_size, num_layers)

# Forward pass
output = model(input_tensor)
print(output.shape)  # (batch_size, seq_len, output_size)
