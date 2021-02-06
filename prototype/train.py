# Inspired by https://github.com/andrewpeng02/transformer-translation

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from prototype.model import Transformer
from Optim import ScheduledOptim
from torch.optim import Adam



def get_training_mask(states, args):
        # Generate the masks
        mask = [[False for _ in range(state.shape[0])] + [True for _ in range(args['maximum_sequence_length'] - state.shape[0])] for state in states]

        # Elongate the state to 'maximum_sequence_length"
        states = [state.tolist() + [args['word_to_id']['<PAD>'] for _ in range(args['maximum_sequence_length'] - state.shape[0])] for state in states]

        return torch.tensor(states), torch.tensor(mask)


def generate_no_peek_mask(target_input):
        # Mask future outputs for the decoder
        mask = (torch.triu(torch.ones(target_input.shape[1], target_input.shape[1])) == 1).T
        mask = mask.float().masked_fill(mask == False, float('-inf')).masked_fill(mask == True, float(0.0))

        return mask


def main():
        # Load the data
        features = np.load('./data/features.npy')
        targets = np.load('./data/target.npy')
        args = np.load('./data/args.npy', allow_pickle=True).item()

        # Convert each feature vector into a vocabulary word
        features = [[args['word_to_id'][word] for word in feature] for feature in features]
        targets = [[args['word_to_id']['SOS']] + [args['word_to_id'][word] for word in target] + [args['word_to_id']['EOS']] for target in targets]

        # Convert into a tensor
        features = torch.LongTensor(features) 
        targets = torch.LongTensor(targets)

        # Prepare into a torch dataset
        dataset = TensorDataset(features,targets) 
        dataloader = DataLoader(dataset, batch_size=32) 

        model_args ={
                'num_embeddings': args['number_of_words'] + 1, 
                'embedding_dim':128,
                'd_model': features.shape[0],
                'nhead':8,
                'num_encoder_layers':6,
                'num_decoder_layers':6,
                'output_dim':args['number_of_words'],
                'max_sequence_length':args['maximum_sequence_length'],
                'positional_dropout':0.1,
                'n_warmup_steps':4000,
        }

        # Define the model
        model = Transformer(model_args).to('cpu')
        model.train()
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        optimizer = ScheduledOptim(
                                Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                                model_args['d_model'], model_args['n_warmup_steps'])

        # optimizer = torch.optim.Adam(model.parameters())

        best_loss = 1e9

        # Iterate through the data
        for i in tqdm(range(1000)):

                total_loss = 0

                for batch_features, batch_labels in dataloader:

                        # Zero out the gradient for this round of updates
                        optimizer.zero_grad()

                        # Transform every input to size 'maximum_sequence_length" and generate the masks for source and target
                        source, source_key_padding_mask = get_training_mask(batch_features, args)
                        target, target_key_padding_mask = get_training_mask(batch_labels, args)

                        # target input should be everrything except EOS
                        target_input, target_key_padding_mask = target[:,:-1], target_key_padding_mask[:,:-1]
                        # target output should be everything but SOS
                        target_output = target[:,1:]

                        # Generate the no peek masks for the decoder
                        target_mask = generate_no_peek_mask(target_input)

                        # Conduct a forward pass of the transformer
                        predictions = model.forward(    source, \
                                                        target_input, \
                                                        tgt_mask = target_mask, \
                                                        src_key_padding_mask=source_key_padding_mask, \
                                                        tgt_key_padding_mask=target_key_padding_mask, \
                                                        memory_key_padding_mask=source_key_padding_mask.clone()
                                                        )


                        # Compare the output of the model to the target
                        number_of_words = predictions.shape[2]
                        loss = criterion(predictions.reshape(-1,number_of_words), target_output.reshape(-1))

                        # Update the model
                        loss.backward()
                        optimizer.step_and_update_lr()

                        total_loss += loss.item()
                        

                if i% 100 == 0 :
                        print(total_loss)
                        if total_loss < best_loss:
                                print("Saving model!")
                                torch.save(model, 'model/transformer.pth')
                                best_loss = total_loss
                                



if __name__ == "__main__":
    main()