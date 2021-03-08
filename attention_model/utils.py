import argparse
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from model import Feedforward, model_a, model_b, model_c, test
from torch.optim import Adam
from torch.utils.data.sampler import Sampler

def init_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s ',  '--save_dir', default='models', help='Path to save the results.')
    parser.add_argument('-m ',  '--model', default='model_c', help='Path of the model.')
    parser.add_argument('-l ',  '--log_directory', default='./log', help='Path of the log file.')
    parser.add_argument('-a ',  '--max_number_of_agents', default=6, type=int, help='Maximum number of agents')
    parser.add_argument('-tb ',  '--train_batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('-vb ',  '--validation_batch_size', default=100, type=int, help='Batch size')
    parser.add_argument('-tt ',  '--test_batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('-td',  '--training_agents', nargs="+", default=[1,2,3,5,6])
    parser.add_argument('-vd',  '--validation_agents', nargs="+", default=[1,2,3,5,6])
    parser.add_argument('-tt',  '--test_agents', nargs="+", default=[4])
    parser.add_argument('-hd ', '--hidden_dimension', default=32, type=int, help='Hidden dimension size')
    parser.add_argument('-e ',  '--embedding_dimension', default=64, type=int, help='Default Embedding dimension')
    parser.add_argument('-ae ', '--action_embedding_dimension', default=64, type=int, help='Action Embedding dimension')
    parser.add_argument('-oe ', '--observation_embedding_dimension', default=64, type=int, help='Observation Embedding dimension')
    parser.add_argument('-se ', '--state_embedding_dimension', default=64, type=int, help='Sstate Embedding dimension')
    parser.add_argument('-t ',  '--training_dataset', default='./data/train.npy', help='Path to training dataset')
    parser.add_argument('-te ', '--test_dataset', default='./data/test.npy', help='Path to test dataset')
    parser.add_argument('-v ',  '--validation_dataset', default='./data/val.npy', help='Path to validation dataset')
    parser.add_argument('-hl ', '--hidden_layers', default=3, type=int, help='Number of hidden layers in the MLP')
    parser.add_argument('-nh ', '--n_heads', default=1, type=int, help='Number of attention heads')
    parser.add_argument('-th ', '--threshold', default=1e-2, type=int, help='Threshold to stop training')


    args = parser.parse_args()   
    assert args.model in ['test', 'model_a', 'model_b', 'model_c', 'Feedforward'], "Not a valid model"

    args.model_dir = f'./{args.save_dir}/{args.model}'
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
        os.mkdir(f'{args.model_dir}/log')
   

    return args


def prepare_dataloader(data, agents, batch_size, args):
    dataloaders = {}
    for n_agent in agents:
        state = data[n_agent]['state']
        action = data[n_agent]['actions']
        target = data[n_agent]['target']
        dataset = Dataset(state, action, target, args)
        dataloaders[n_agent] = DataLoader(dataset, batch_size = batch_size, shuffle=True, drop_last = True)
    return dataloaders



def initialize_dataloader(args, subset=None):
    training_data = np.load(args.training_dataset, allow_pickle=True).item()
    validation_data = np.load(args.validation_dataset, allow_pickle=True).item()
    test_data = np.load(args.test_dataset, allow_pickle=True).item()

    args.observation_dimension = training_data[1]['state'][0].shape[1]
    args.action_dimension = training_data[1]['actions'][0].shape[1]
    args.state_dimension = args.max_number_of_agents * args.observation_dimension

    train_dataloaders = prepare_dataloader(training_data, args.training_agents, args.train_batch_size, args)
    val_dataloaders = prepare_dataloader(validation_data, args.validation_agents, args.validation_batch_size, args)
    test_dataloaders = prepare_dataloader(test_data, args.test_agents, args.test_batch_size, args)


    # Truncate the dataset
    # if subset is not None:
    #     state_features = state_features[:subset]
    #     action_features = action_features[:subset]
    #     targets = targets[:subset]

    # Model parametersn

    # # Prepare into a torch dataset
    # training_dataset = Dataset(state_features, action_features, targets, args) 
    # training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) 
    
    # validation_dataset = Dataset(val_state_features, val_action_features, val_targets, args) 
    # validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, drop_last = True) 

    # test_dataset = Dataset(test_state_features, test_action_features, test_targets, args) 
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) 

    return train_dataloaders, val_dataloaders, test_dataloaders, args


class Dataset(torch.utils.data.Dataset):
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, states, actions, targets, args):
        'Initialization'
        self.states = states
        self.actions= actions
        self.targets = targets
        self.args = args

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.states)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        if self.args.model == 'model_a':
            state = np.vstack((self.states[index],np.zeros((self.args.max_number_of_agents, self.args.observation_dimension))))[:self.args.max_number_of_agents,:]
            action = np.vstack((self.actions[index],np.zeros((self.args.max_number_of_agents, self.args.action_dimension))))[:self.args.max_number_of_agents,:]
            target = np.vstack((self.targets[index],np.zeros((self.args.max_number_of_agents, self.args.observation_dimension))))[:self.args.max_number_of_agents,:]

            # Flatten
            state   = state.reshape(1,-1)
            target  = target.reshape(1,-1)

        elif self.args.model == 'Feedforward':
            state = np.vstack((self.states[index],np.zeros((self.args.max_number_of_agents, self.args.observation_dimension))))[:self.args.max_number_of_agents,:]
            action = np.vstack((self.actions[index],np.zeros((self.args.max_number_of_agents, self.args.action_dimension))))[:self.args.max_number_of_agents,:]
            target = np.vstack((self.targets[index],np.zeros((self.args.max_number_of_agents, self.args.observation_dimension))))[:self.args.max_number_of_agents,:]
        
        elif self.args.model == 'model_b':
            state = self.states[index]
            action = self.actions[index]
            target = self.targets[index]
        
        elif self.args.model == 'model_c':
            state = self.states[index]
            action = self.actions[index]
            target = self.targets[index]

            state = state[0].reshape(1,-1)
            target = target[0].reshape(1,-1)
        
        else:
            state, action, target = None, None, None

        return state, action, target


def initialize_model(args):
    model_path = f'{args.model_dir}/model.pth'
    if os.path.exists(model_path):
        f = open(f'{args.model_dir}/log/log.txt', 'w')
        f.write("Loading Model.\n")
        f.close()
        print(f"Loading model from {model_path}")
        model = torch.load(f'{model_path}')
    elif args.model == 'Feedforward':
        model = Feedforward(args)
    elif args.model == 'model_a':
        model = model_a(args)
    elif args.model == 'model_b':
        model = model_b(args)
    elif args.model =='model_c':
        model = model_c(args)
    elif args.model == 'test':
        model = test(args)
    else:
        model = None
        assert(False, "Invalid Entry")
    
    criterion = torch.nn.L1Loss()
    optimizer = Adam(model.parameters())

    return model, criterion, optimizer


def evaluate(model, dataloaders, criterion):
        total_loss = 0
        model.eval()
        n_batches = 0.0
        with torch.no_grad():
            for dataloader in dataloaders.values():
                for observations, actions, target in dataloader:
                        prediction = model.forward(observations, actions)

                        prediction = torch.Tensor(prediction.flatten())
                        target = torch.Tensor(target.flatten().float())
                        loss = criterion(prediction, target)

                        total_loss += loss.item()
                        n_batches += 1.0
        model.train()
        total_loss = total_loss/ n_batches
        return total_loss


def log(epoch, args, validation_loss=None, training_loss=None, test_loss = None):

        if test_loss is not None:
                f = open(f'{args.model_dir}/log/log.txt', 'a')
                f.write(f'Test loss: {test_loss}')
                print(f'Test loss: {test_loss}')
                f.close()
        else:
                # Save to log and print to output
                f = open(f'{args.model_dir}/log/log.txt', 'a')
                f.write(f'iteration {epoch}s Training loss: {training_loss}, validation loss: {validation_loss}\n')
                print(f'iteration {epoch}s Training loss: {training_loss}, validation loss: {validation_loss}\n')
                f.close()


def save_model(model, args, epoch):
        f = open(f'{args.model_dir}/log/log.txt', 'a')
        f.write(f"Saving model on iteration {epoch}\n")
        print(f"Saving model on iteration {epoch}\n")
        torch.save(model, f"{args.model_dir}/model.pth")
        f.close()