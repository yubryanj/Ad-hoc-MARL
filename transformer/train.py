import argparse
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformer.model import Transformer, Feedforward
from torch.optim import Adam
import os


def prepare_data(dataset):
        features = []
        targets = []
        for sample in dataset:
                initial_state = np.concatenate(sample[0]).ravel()
                action = np.concatenate(sample[1]).ravel()
                next_state = np.concatenate(sample[2]).ravel()

                features.append(np.concatenate((initial_state,action)))
                targets.append(next_state)

        return features, targets
                

def main():

        parser = argparse.ArgumentParser(description=None)
        parser.add_argument('-s', '--scenario', default='data_generate.py', help='Path of the scenario Python script.')
        parser.add_argument('-m ', '--model', default='./model/feedforward.pth', help='Path of the model.')
        args = parser.parse_args()

        # Load the data
        features = np.load('./data/features.npy', allow_pickle=True)
        targets = np.load('./data/targets.npy', allow_pickle=True)

        args.input_dimension = features[0].shape[0]
        args.output_dimension = targets[0].shape[0]
        args.hidden_dimension = 128
        args.number_of_layers = 3

        # Convert into a tensor
        features = torch.Tensor(features) 
        targets = torch.Tensor(targets)

        # Prepare into a torch dataset
        dataset = TensorDataset(features,targets) 
        dataloader = DataLoader(dataset, batch_size=256) 

        if os.path.isfile(args.model):
                print(f'Loading model.')
                model = torch.load(args.model)
        else:
                model = Feedforward(args)

        model.train()
        criterion = torch.nn.MSELoss()
        optimizer = Adam(model.parameters())

        best_loss = 1e9

        # Iterate through the data
        for i in tqdm(range(20000)):

                total_loss = 0

                for batch_features, batch_labels in dataloader:

                        # Zero out the gradient for this round of updates
                        optimizer.zero_grad()

                        # Conduct a forward pass of the transformer
                        predictions = model.forward(batch_features.float())

                        # Compare the output of the model to the target
                        loss = criterion(predictions, batch_labels)

                        # Update the model
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        

                if i% 50 == 0 :
                        print(f'iteration {i}s total loss: {total_loss}')
                        if total_loss < best_loss:
                                print("Saving model!")
                                torch.save(model, 'model/feedforward.pth')
                                best_loss = total_loss
                                

if __name__ == "__main__":
    main()