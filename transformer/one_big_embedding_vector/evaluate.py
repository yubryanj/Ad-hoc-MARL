import argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model import Transition_Model
import os

                

def main():

        parser = argparse.ArgumentParser(description=None)
        parser.add_argument('-m ', '--model', default='./model/model_01.pth', help='Path of the model.')
        args = parser.parse_args()

        # Load the data
        features = np.load('./data/features.npy', allow_pickle=True)
        targets = np.load('./data/targets.npy', allow_pickle=True)

        args.input_dimension = features[0].shape[0]
        args.output_dimension = targets[0].shape[0]
        args.hidden_dimension = 128
        args.number_of_layers = 3

        # Convert into a tensor
        features = torch.Tensor(features[:100]) 
        targets = torch.Tensor(targets[:100])

        # Prepare into a torch dataset
        dataset = TensorDataset(features,targets) 
        dataloader = DataLoader(dataset, batch_size=256) 

        if os.path.isfile(args.model):
            print(f'Loading model.')
            model = torch.load(args.model)
        else:
            assert(False)

        # Conduct a forward pass of the transformer
        predictions = model.forward(features.unsqueeze(1).float())

        # Compare the output of the model to the target
        loss = torch.nn.MSELoss(predictions, targets.unsqueeze(1))

        print(f'iteration {i}s total loss: {loss}')
                        

if __name__ == "__main__":
    main()