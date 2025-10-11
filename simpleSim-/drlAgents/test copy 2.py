import torch
import torch.nn as nn
import torch.optim as optim
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)  # input_dim = 7853=len(de_vocab)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))  # src=13,1   embedded=13,1,256
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded) # output_dim=5893=len(en_vocab)  outputs=torch.Size([13, 1, 512]),# hidden=cell=torch.Size([2, 1, 512])
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
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size , hidden dim]
        input = input.unsqueeze(0)  #  torch.Size([1, 1])
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell)) # torch.Size([1, 1, 512]) torch.Size([2, 1, 512]) torch.Size([2, 1, 512])
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0)) # torch.Size([1, 5893])
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

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1] # 1
        trg_length = trg.shape[0] # 15
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device) # 15,1,5893
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]    # <sos> tokens  [2] 
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell) # [2]   torch.Size([2, 1, 512]) torch.Size([2, 1, 512])
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output  #从1开始，因为0是<sos>  torch.Size([1, 5893])
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs

input_dim = 7853  # ['<unk>'：0, '<pad>'：1, '<sos>', '<eos>', '.', 'ein', 'einem', 'in', 'eine', ',']
output_dim = 5893   # ['<unk>', '<pad>', '<sos>', '<eos>', 'a', '.', 'in', 'the', 'on', 'man']
pad_index = -100    # en_vocab[pad_token]  pad_token="<pad>",设置为-100
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)

model = Seq2Seq(encoder, decoder, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
# 我们的损失函数计算每个标记的平均损失，但是通过将 <pad> 标记的索引作为 ignore_index 参数传递，只要目标标记是填充标记，我们就会忽略损失。
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

# train_data[0]
# {'en': 'Two young, White males are outside near many bushes.',
#  'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}
# 'en_tokens': ['<sos>',
#   'two',
#   'young',
#   ',',
#   'white',
#   'males',
#   'are',
#   'outside',
#   'near',
#   'many',
#   'bushes',
#   '.',
#   '<eos>'],
#  'de_tokens': ['<sos>',
#   'zwei',
#   'junge',
#   'weiße',
#   'männer',
#   'sind',
#   'im',
#   'freien',
#   'in',
#   'der',
#   'nähe',
#   'vieler',
#   'büsche',
#   '.',
#   '<eos>']}
# 

clip = 1.0
teacher_forcing_ratio = 0.5
model.train()
epoch_loss = 0
en_ids = [torch.tensor([2,16,24, 15,25,778,17,57,80,202,1312,   5,   3])]
de_ids = [torch.tensor([2,18,26,253,30, 84,20,88, 7, 15, 110,764,3171,4,3])]
batch_en_ids = nn.utils.rnn.pad_sequence(en_ids, padding_value=pad_index)
batch_de_ids = nn.utils.rnn.pad_sequence(de_ids, padding_value=pad_index)
src = batch_en_ids.to(device)
trg = batch_de_ids.to(device)
# src = [src length, batch size]
# trg = [trg length, batch size]
optimizer.zero_grad()
output = model(src, trg, teacher_forcing_ratio)
# output = [trg length, batch size, trg vocab size]
output_dim = output.shape[-1]
output = output[1:].view(-1, output_dim)
# output = [(trg length - 1) * batch size, trg vocab size]
trg = trg[1:].view(-1)
# trg = [(trg length - 1) * batch size]
loss = criterion(output, trg)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
optimizer.step()
epoch_loss += loss.item()

