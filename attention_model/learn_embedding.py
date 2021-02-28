import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import attention_v2, Feedforward, attention_v1
from torch.optim import Adam
import os


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, states, actions, targets, action_to_id, args):
        'Initialization'
        self.states = states
        self.actions= actions
        self.targets = targets
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
        elif self.args.model == 'attention_v2':
                state = self.states[index]
                action = np.array([self.action_to_id[tuple(i)] for i in self.actions[index]])
                target = self.targets[index]
        elif self.args.model == 'attention_v1':
                state = self.states[index]
                action = np.array([self.action_to_id[tuple(i)] for i in self.actions[index]])
                target = self.targets[index]
        else:
                assert False, "Not a valid model"

        return state, action, target


def initialize_model(args):
        if os.path.exists(f"./models/{args.model}.pth"):
                f = open(f'{args.log_directory}/{args.model}_log.txt', 'w')
                f.write("Loading Model.\n")
                f.close()
                print(f"Loading model from './models/{args.model}.pth")
                model = torch.load(f"./models/{args.model}.pth")
        if args.model == 'feedforward':
                args.hidden_dimension = args.hidden_dimension
                args.number_of_layers = 3
                model = Feedforward(args)
        elif args.model == 'attention_v1':
                model = attention_v1(args)
        elif args.model == 'attention_v2':
                model = attention_v2(args)
        else:
                model = None
                assert(False, "Invalid Entry")
        

        criterion = torch.nn.MSELoss()
        optimizer = Adam(model.parameters())

        return model, criterion, optimizer


def initialize_dataloader(args):
        # Load the data
        training_data = np.load('./data/embeddings_dataset_variable_agents.npy', allow_pickle=True).item()
        validation_data = np.load('./data/embeddings_dataset_variable_agents.npy', allow_pickle=True).item()

        # Extract the features and targets
        state_features, action_features, targets = training_data['state_features'], training_data['action_features'], training_data['targets']
        val_state_features, val_action_features, val_targets = validation_data['state_features'], validation_data['action_features'], validation_data['targets']
        
        
        # Extract the mapping dictionaries
        action_to_id = training_data['action_to_id']

        # Model parameters
        args.number_of_actions = len(action_to_id)
        args.state_input_dimension = state_features[0].shape[1]
        args.action_input_dimension = action_features[0].shape[1]
        args.output_dimension = targets[0].flatten().shape[0]

        # Prepare into a torch dataset
        training_dataset = Dataset(state_features, action_features, targets, action_to_id, args) 
        training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True) 
        validation_dataset = Dataset(val_state_features, val_action_features, val_targets, action_to_id, args) 
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True) 

        return training_dataloader, validation_dataloader, args



def main():

        # Assumes the "UNK" will learn a mapping which will aid predictions of unseen states
        # Can update the model to only create embeddings for frequently occuring states
        # Can truncate the states to smaller decimals, so there is increased overlaps (i.e. 1.53 and 1.59 both maps to 1.5)

        parser = argparse.ArgumentParser(description=None)
        parser.add_argument('-m ', '--model', default='feedforward', help='Path of the model.')
        parser.add_argument('-l ', '--log_directory', default='./log', help='Path of the log file.')
        parser.add_argument('-a ', '--max_number_of_agents', default=6, help='Maximum number of agents')
        parser.add_argument('-n ', '--n_epochs', default=100, help='Number of epochs to run')
        parser.add_argument('-b ', '--batch_size', default=256, help='Batch size')
        parser.add_argument('-h ', '--hidden_dimension', default=512, help='Hidden dimension size')
        parser.add_argument('-e ', '--embedding_dimension', default=512, help='Embedding dimension size')
        args = parser.parse_args()      

        assert args.model in ['attention_v1','attention_v2','feedforward'], "Not a valid model"

        # Open log file
        f = open(f'{args.log_directory}/{args.model}_log.txt', 'w')
        f.write("Starting training.\n")
        f.close()

        # Prepare the dataloader
        training_dataloader, validation_dataloader, args = initialize_dataloader(args)
        
        # Prepare the model
        model, criterion, optimizer = initialize_model(args)

        best_loss = 1e9
        training_losses = []
        validation_losses = []
        model.train()

        # Iterate through the data
        for i in tqdm(range(args.n_epochs)):

                total_loss = 0

                for states, actions, target in training_dataloader:

                        batch_size = target.shape[0]

                        # Zero out the gradient for this round of updates
                        optimizer.zero_grad()

                        # Conduct a forward pass of the transformer
                        predictions = model.forward(states, actions)

                        # Compare the output of the model to the target
                        loss = criterion(predictions.float(), target.reshape(batch_size,-1).float())

                        # Update the model
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                # Check against the validation dataset
                model.eval()
                with torch.no_grad():
                        validation_loss = 0
                        for v_states, v_actions, v_target in  validation_dataloader:
                                batch_size = v_target.shape[0]
                                v_predictions = model.forward(v_states, v_actions)
                                loss = criterion(v_predictions.float(), v_target.reshape(batch_size,-1).float())
                                validation_loss += loss.item()

                        if validation_loss < best_loss:
                                f = open(f'{args.log_directory}/{args.model}_log.txt', "a")
                                f.write(f"Saving model on iteration {i}\n")
                                print(f"Saving model on iteration {i}\n")
                                torch.save(model, f"./models/{args.model}.pth")
                                f.close()
                                best_loss = validation_loss

                model.train()
                
                # Save the losses
                training_losses.append(total_loss)
                validation_losses.append(validation_loss)

                # Save to log and print to output
                f = open(f'{args.log_directory}/{args.model}_log.txt', "a")
                f.write(f'iteration {i}s total loss: {total_loss}, validation loss: {validation_loss}\n')
                print(f'iteration {i}s total loss: {total_loss}, validation loss: {validation_loss}\n')
                f.close()

        # Save the losses
        np.save(f'{args.log_directory}/{args.model}_training_losses.npy', training_losses)
        np.save(f'{args.log_directory}/{args.model}_validation_losses.npy', validation_losses)


if __name__ == "__main__":
    main()