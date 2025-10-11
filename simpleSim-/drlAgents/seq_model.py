import torch
import torch.nn as nn
import torch.optim as optim
import random

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = 1  # embedding_dim=256
        # self.embedding = nn.Embedding(input_dim, embedding_dim)  # input_dim = 7853=len(de_vocab)
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        # embedded = self.dropout(self.embedding(src))  # src=13,1   embedded=13,1,256
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(src) # output_dim=5893=len(en_vocab)  outputs=torch.Size([13, 1, 512]),# hidden=cell=torch.Size([2, 1, 512])
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        # self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size , hidden dim]
        # input = input.unsqueeze(0)  #  torch.Size([1, 1])
        # # input = [1, batch size]
        # embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(input, (hidden, cell)) # torch.Size([1, 1, 512]) torch.Size([2, 1, 512]) torch.Size([2, 1, 512])
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output) # torch.Size([1, 5893])
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        # 解码器初始化输入为零序列 (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = src.size()
        input = torch.zeros(batch_size, seq_len, hidden.size(2))
        # input = [batch size]
        output, hidden, cell = self.decoder(input, hidden, cell) # [2]   torch.Size([2, 1, 512]) torch.Size([2, 1, 512])
        return output
    
    
