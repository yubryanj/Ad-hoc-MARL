import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Learn_Embeddings_with_attention, Feedforward
from torch.optim import Adam
import os


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, states, actions, targets, state_to_id, action_to_id, args):
        'Initialization'
        self.states = states
        self.actions= actions
        self.targets = targets
        self.state_to_id = state_to_id
        self.action_to_id = action_to_id
        self.args = args

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.states)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        if self.args.model =='feedforward':
                state = self.states[index]
                action = self.actions[index]
                target = self.targets[index]        
        elif self.args.model == 'attention':
                state = self.states[index]
                action = self.action_to_id[tuple(self.actions[index])]
                target = self.targets[index]
        else:
                assert False, "Not a valid model"

        return state, action, target

def main():

        # Transfer Learning
        # Take the weights learned from the trained linear classifier and load it into the the deeper model

        # Assumes the "UNK" will learn a mapping which will aid predictions of unseen states
        # Can update the model to only create embeddings for frequently occuring states
        # Can truncate the states to smaller decimals, so there is increased overlaps (i.e. 1.53 and 1.59 both maps to 1.5)

        parser = argparse.ArgumentParser(description=None)
        parser.add_argument('-m ', '--model', default='feedforward', help='Path of the model.')
        parser.add_argument('-l ', '--log_directory', default='./log', help='Path of the log file.')

        args = parser.parse_args()      

        assert args.model in ['attention','feedforward'], "Not a valid model"

        # Open log file
        f = open(f'{args.log_directory}/{args.model}_log.txt', 'w')
        f.write("Starting training.\n")
        f.close()

        # Load the data
        data = np.load('./data/embeddings_dataset.npy', allow_pickle=True).item()

        # Extract the features and targets
        state_features, action_features, targets = data['state_features'], data['action_features'], data['targets']
        
        # Extract the mapping dictionaries
        state_to_id, action_to_id = data['state_to_id'], data['action_to_id']

        # Model parameters
        args.number_of_states = len(state_to_id)
        args.number_of_actions = len(action_to_id)
        args.embedding_dimension = 512
        args.state_input_size = state_features[0].shape[0]
        args.action_input_size = action_features[0].shape[0]
        args.output_size = targets[0].shape[0]

        # Prepare into a torch dataset
        dataset = Dataset(state_features, action_features, targets, state_to_id, action_to_id, args) 
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True) 

        if os.path.exists(f"./models/{args.model}.pth"):
                f = open(f'{args.log_directory}/{args.model}_log.txt', 'w')
                f.write("Loading Model.\n")
                f.close()
                model = torch.load(f"./models/{args.model}.pth")
        if args.model == 'feedforward':
                args.hidden_dimension = 512
                args.number_of_layers = 3
                model = Feedforward(args)
        elif args.model == 'attention':
                model = Learn_Embeddings_with_attention(args)
        else:
                model = None
                assert(False, "Invalid Entry")


        # MSE if we're doing a regression to the next state
        criterion = torch.nn.MSELoss()
        optimizer = Adam(model.parameters())

        best_loss = 1e9
        losses = []
        model.train()

        # Iterate through the data
        for i in tqdm(range(50)):

                total_loss = 0

                for states, actions, target in dataloader:

                        # Zero out the gradient for this round of updates
                        optimizer.zero_grad()

                        # Conduct a forward pass of the transformer
                        predictions = model.forward(states, actions)

                        # Compare the output of the model to the target
                        loss = criterion(predictions.squeeze().float(), target.float())

                        # Update the model
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
        

                # if i% 50 == 0 :
                f = open(f'{args.log_directory}/{args.model}_log.txt', "a")
                f.write(f'iteration {i}s total loss: {total_loss}\n')
                f.close()

                # Save the model
                if total_loss < best_loss:
                        f = open(f'{args.log_directory}/{args.model}_log.txt', "a")
                        f.write(f"Saving model on iteration {i}\n")
                        torch.save(model, f"./models/{args.model}.pth")
                        f.close()
                        best_loss = total_loss

                losses.append(total_loss)

        # Save the losses
        np.save(f'{args.log_directory}/{args.model}_losses.npy', losses)

if __name__ == "__main__":
    main()