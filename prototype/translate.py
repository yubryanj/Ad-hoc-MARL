# Modeled after https://github.com/andrewpeng02/transformer-translation

import pickle
from einops import rearrange
import numpy as np

import torch


def main():
    model = torch.load('model/transformer.pth')
    args = np.load('./data/args.npy', allow_pickle=True).item()
    features = np.load('./data/features.npy')
    targets = np.load('./data/target.npy')[0]


    current_state = features[0]
    current_target = targets[0]

    # Tokenize input
    # current_state = input('Please the current state: ')
    tokenized_current_state = tokenize(current_state, args)

    next_state = [args['word_to_id']['SOS']]

    while int(next_state[-1]) != args['word_to_id']['EOS'] and len(next_state)<args['maximum_sequence_length']:
        output = forward_model(model, tokenized_current_state, next_state)
        values, indices = torch.topk(output, 5)
        next_state.append(int(indices.flatten()[0]))

    print(detokenize(next_state, args))


def forward_model(model, src, tgt):
    src = torch.tensor(src).unsqueeze(0).long()
    tgt = torch.tensor(tgt).unsqueeze(0)
    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    output = model.forward(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=None)

    return output.squeeze(0)


def tokenize(initial_state, args):

    return [args['word_to_id'][token] for token in initial_state]


def detokenize(state, args):
    return [args['id_to_word'][id] for id in state]

def gen_nopeek_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


if __name__ == "__main__":
    main()
