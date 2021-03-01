import argparse
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from model import attend_over_actions, Feedforward, attend_over_state_and_actions
from torch.optim import Adam

def init_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-m ', '--model', default='central_model', help='Path of the model.')
    parser.add_argument('-l ', '--log_directory', default='./log', help='Path of the log file.')
    parser.add_argument('-a ', '--max_number_of_agents', default=6, help='Maximum number of agents')
    parser.add_argument('-b ', '--batch_size', default=512, help='Batch size')
    parser.add_argument('-h ', '--hidden_dimension', default=512, help='Hidden dimension size')
    parser.add_argument('-e ', '--embedding_dimension', default=512, help='Embedding dimension size')
    parser.add_argument('-t ', '--training_dataset', default='./data/training_v1.npy', help='Path to training dataset')
    parser.add_argument('-te ', '--test_dataset', default='./data/test_v1.npy', help='Path to test dataset')
    parser.add_argument('-v ', '--validation_dataset', default='./data/validation_v1.npy', help='Path to validation dataset')

    args = parser.parse_args()   
    assert args.model in ['attend_over_state_and_actions', 'attend_over_actions', 'central_model', 'Feedforward'], "Not a valid model"
   
    return args


def initialize_dataloader(args, pad_targets=True):
    # Load the data
    training_data = np.load(args.training_dataset, allow_pickle=True).item()
    validation_data = np.load(args.validation_dataset, allow_pickle=True).item()
    test_data = np.load(args.test_dataset, allow_pickle=True).item()

    # Extract the features and targets
    state_features, action_features, targets = training_data['state_features'], training_data['action_features'], training_data['targets']
    val_state_features, val_action_features, val_targets = validation_data['state_features'], validation_data['action_features'], validation_data['targets']
    test_state_features, test_action_features, test_targets = test_data['state_features'], test_data['action_features'], test_data['targets']

    # Extract the mapping dictionaries
    action_to_id = training_data['action_to_id']

    # Model parameters
    args.number_of_actions = len(action_to_id)
    args.observation_input_dimension = state_features[0].shape[1]
    args.action_input_dimension = action_features[0].shape[1]
    args.output_dimension = args.max_number_of_agents * args.observation_input_dimension

    # Prepare into a torch dataset
    training_dataset = Dataset(state_features, action_features, targets, action_to_id, args) 
    training_sampler = Variable_Length_Sampler(state_features, args)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, sampler=training_sampler) 
    
    validation_dataset = Dataset(val_state_features, val_action_features, val_targets, action_to_id, args) 
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True) 

    test_dataset = Dataset(test_state_features, test_action_features, test_targets, action_to_id, args) 
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True) 

    return training_dataloader, validation_dataloader, test_dataloader, args


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

        if self.args.model =='Feedforward':
            state = self.states[index]
            action = self.actions[index]
            target = self.targets[index]        
        elif self.args.model == 'central_model':
            state = np.vstack((self.states[index],np.zeros((self.args.max_number_of_agents, self.args.observation_input_dimension))))[:self.args.max_number_of_agents,:]
            action = np.array([self.action_to_id[tuple(i)] for i in self.actions[index]]+[6 for _ in range(self.args.max_number_of_agents)])[:self.args.max_number_of_agents]
            target = np.vstack((self.targets[index],np.zeros((self.args.max_number_of_agents, self.args.observation_input_dimension))))[:self.args.max_number_of_agents,:]
        elif self.args.model in ['attend_over_state_and_actions','attend_over_actions']:
            state = self.states[index]
            action = np.array([self.action_to_id[tuple(i)] for i in self.actions[index]])
            target = np.vstack((self.targets[index],np.zeros((self.args.max_number_of_agents, self.args.observation_input_dimension))))[:self.args.max_number_of_agents,:]
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
    if args.model == 'Feedforward':
        args.hidden_dimension = args.hidden_dimension
        args.number_of_layers = 3
        model = Feedforward(args)
    elif args.model in ['attend_over_state_and_actions','central_model']:
        model = attend_over_state_and_actions(args)
    elif args.model == 'attend_over_actions':
        model = attend_over_actions(args)
    else:
        model = None
        assert(False, "Invalid Entry")
    
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters())

    return model, criterion, optimizer



from torch.utils.data.sampler import Sampler
class Variable_Length_Sampler(Sampler):

    def __init__(self, data, args):
        self.args = args
        self.data = self.process_data(data)
        self.n_samples = len(data)

    def process_data(self, data):
        processed_data = {}
        for i in range(1, self.args.max_number_of_agents + 1):
            processed_data[i] = []

        for i, sample in enumerate(data):
            processed_data[len(sample)].append(i)        
        return processed_data

    def __iter__(self):
        subset = []
        while len(subset) == 0:
            # Sample the number of agents for this iteration
            number_of_agents = np.random.randint(1, self.args.max_number_of_agents)
            subset = self.data[number_of_agents]
            print(f"Training on subset with {number_of_agents} agents")
        
        # Shuffle the samples
        np.random.shuffle(subset)

        return iter(subset)

    def __len__(self):
        return self.n_samples


