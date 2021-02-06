import math
import torch
from torch import nn

# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        self.embedding_dim = args['embedding_dim']

        self.source_embedding = nn.Embedding(args['num_embeddings'], args['embedding_dim'])
        self.target_embedding = nn.Embedding(args['num_embeddings'], args['embedding_dim'])
        self.transformer = nn.Transformer(args['embedding_dim'],args['nhead'],args['num_encoder_layers'],args['num_decoder_layers'])
        self.linear = nn.Linear(args['embedding_dim'], args['output_dim'])
        self.positional_encoder = PositionalEncoding(args['embedding_dim'], args['positional_dropout'], args['max_sequence_length'])


    def forward(self, source, target, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=None):
        source_embedding = self.positional_encoder(self.source_embedding(source.long().T) * math.sqrt(self.embedding_dim))
        target_embedding = self.positional_encoder(self.target_embedding(target.long().T) * math.sqrt(self.embedding_dim))

        output = self.transformer(  source_embedding, \
                                    target_embedding, \
                                    tgt_mask=tgt_mask, \
                                    src_key_padding_mask=src_key_padding_mask, \
                                    tgt_key_padding_mask=tgt_key_padding_mask, \
                                    memory_key_padding_mask=memory_key_padding_mask
                                    )

        return self.linear(output)

