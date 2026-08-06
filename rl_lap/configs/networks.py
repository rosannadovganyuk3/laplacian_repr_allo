import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RNN_CELLS = {
    'LSTM': nn.LSTM,
    'GRU': nn.GRU,
    'RNN': nn.RNN,
}


class RNN(nn.Module):
    def __init__(self, input_shape, n_layers, n_units, rnn_type='LSTM'):
        super().__init__()
        # input_shape for HardMaze is usually (C, H, W)
        # We flatten this for a pure RNN
        self.input_dim = int(np.prod(input_shape))

        # 1. Initial projection layer to get pixels into a latent space
        self.feature_projection = nn.Linear(self.input_dim, n_units)

        # 2. The Recurrent Backbone (LSTM/GRU/RNN, selected by rnn_type)
        # batch_first=True means input shape is (Batch, Seq, Feature)
        rnn_cls = RNN_CELLS[rnn_type]
        self.rnn = rnn_cls(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, x, hidden=None):
        # 1. Handle the dimensionality safety check
        if x.dim() == 4:
            x = x.unsqueeze(1)

        batch_size, seq_len, C, H, W = x.size()

        # 2. CHANGE THIS LINE: Use .reshape instead of .view
        # This handles the memory "non-contiguity" issue automatically
        x = x.reshape(batch_size, seq_len, -1)

        # 3. Project to n_units
        x = F.relu(self.feature_projection(x))

        # RNN pass
        output, hidden = self.rnn(x, hidden)

        return output[:, -1, :], hidden

class ReprNetRNN(nn.Module):
    def __init__(self, input_shape, n_layers, n_units, d, rnn_type='LSTM'):
        super().__init__()
        self.encoder = RNN(input_shape, n_layers, n_units, rnn_type=rnn_type)
        # The RNN outputs n_units, so we map that to our embedding dimension d
        self.out_layer = nn.Linear(n_units, d)

    def forward(self, x, hidden=None):
        h, hidden = self.encoder(x, hidden)
        return self.out_layer(h)

class DiscreteQNetRNN(nn.Module):
    def __init__(self, input_shape, n_actions, n_layers, n_units, rnn_type='LSTM'):
        super().__init__()
        self.encoder = RNN(input_shape, n_layers, n_units, rnn_type=rnn_type)
        self.out_layer = nn.Linear(n_units, n_actions)

    def forward(self, x, hidden=None):
        h, hidden = self.encoder(x, hidden)
        return self.out_layer(h)