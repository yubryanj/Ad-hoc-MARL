import math
import torch
from torch import nn


class Learn_Embeddings(nn.Module):
    def __init__(self, args):
        super(Learn_Embeddings, self).__init__()

        self.state_embedding = nn.Embedding(args.number_of_states, args.embedding_dimension)
        self.action_embedding = nn.Embedding(args.number_of_actions, args.embedding_dimension)
        self.linear = nn.Linear(args.embedding_dimension * 2,args.output_size)

    def forward(self, state, action):
        embedded_state = self.state_embedding(state)
        embedded_action = self.action_embedding(action)

        features_vector = torch.cat((embedded_state,embedded_action), axis=1)

        predictions = self.linear(features_vector)
        
        return predictions


class Feedforward(nn.Module):

    def __init__(self, args):
        super(Feedforward, self).__init__()

        layers_dimension        = [args.hidden_dimension for _ in range(args.number_of_layers)]
        layers_dimension[0]     = args.input_dimension
        layers_dimension[-1]    = args.output_dimension

        self.layers = nn.ModuleList([nn.Linear(layers_dimension[i],layers_dimension[i+1]) \
                                            for i in range(args.number_of_layers-1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition_Model(nn.Module):

    def __init__(self, args):
        super(Transition_Model, self).__init__()

        # Dimensino to the transformer should be (batchsize, number of features, embedding dimension)
        # Then, need to modify this dataset!
        
        self.embedding = nn.Linear(args.input_dimension, args.hidden_dimension)
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_dimension, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output = nn.Linear(args.hidden_dimension, args.output_dimension)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.output(x)
        return x


class Transformer(nn.Module):

    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim

        self.source_embedding = nn.Embedding(args.num_embeddings, args.embedding_dim)
        self.target_embedding = nn.Embedding(args.num_embeddings, args.embedding_dim)
        self.transformer = nn.Transformer(args.embedding_dim,args.nhead,args.n_encoder_layers,args.n_decoder_layers)
        self.linear = nn.Linear(args.embedding_dim, args.output_dim)


    def forward(self, source, target, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=None):
        source_embedding = self.source_embedding(source.long().T) * math.sqrt(self.embedding_dim)
        target_embedding = self.target_embedding(target.long().T) * math.sqrt(self.embedding_dim)

        output = self.transformer(  source_embedding, \
                                    target_embedding, \
                                    tgt_mask=tgt_mask, \
                                    src_key_padding_mask=src_key_padding_mask, \
                                    tgt_key_padding_mask=tgt_key_padding_mask, \
                                    memory_key_padding_mask=memory_key_padding_mask
                                    )

        return self.linear(output)

