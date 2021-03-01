import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils import initialize_dataloader, initialize_model


def main():

        parser = argparse.ArgumentParser(description=None)
        parser.add_argument('-m ', '--model', default='central_model', help='Path of the model.')
        parser.add_argument('-l ', '--log_directory', default='./log', help='Path of the log file.')
        parser.add_argument('-a ', '--max_number_of_agents', default=6, help='Maximum number of agents')
        parser.add_argument('-n ', '--n_epochs', default=100, help='Number of epochs to run')
        parser.add_argument('-b ', '--batch_size', default=1, help='Batch size')
        parser.add_argument('-h ', '--hidden_dimension', default=512, help='Hidden dimension size')
        parser.add_argument('-e ', '--embedding_dimension', default=512, help='Embedding dimension size')
        parser.add_argument('-t ', '--training_dataset', default='./data/training_v1.npy', help='Path to training dataset')
        parser.add_argument('-te ', '--test_dataset', default='./data/test_v1.npy', help='Path to test dataset')
        parser.add_argument('-v ', '--validation_dataset', default='./data/validation_v1.npy', help='Path to validation dataset')

        args = parser.parse_args()      

        assert args.model in ['attend_over_state_and_actions','attend_over_actions','central_model','Feedforward'], "Not a valid model"

        # Open log file
        f = open(f'{args.log_directory}/{args.model}_log.txt', 'w')
        f.write("Starting training.\n")
        f.close()

        # Prepare the dataloader
        training_dataloader, validation_dataloader, test_dataloader, args = initialize_dataloader(args)
        
        # Prepare the model
        model, criterion, optimizer = initialize_model(args)

        best_validation_loss = float('inf')
        training_losses = []
        validation_losses = []
        model.train()
        epoch = 0

        # Iterate through the data
        while best_validation_loss > 0.10:

                epoch += 1
                total_loss = 0

                for observations, actions, target in tqdm(training_dataloader):

                        batch_size = target.shape[0]

                        # Zero out the gradient for this round of updates
                        optimizer.zero_grad()

                        # Conduct a forward pass of the transformer
                        predictions = model.forward(observations, actions)

                        # Compare the output of the model to the target
                        loss = criterion(predictions.reshape(batch_size,-1).float(), target.reshape(batch_size,-1).float())

                        # Update the model
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                # Check against the validation dataset
                model.eval()
                with torch.no_grad():
                        validation_loss = 0
                        for v_observations, v_actions, v_target in  validation_dataloader:
                                batch_size = v_target.shape[0]
                                v_predictions = model.forward(v_observations, v_actions)
                                v_loss = criterion(v_predictions.float(), v_target.reshape(batch_size,-1).float())
                                validation_loss += v_loss.item()

                        if validation_loss < best_validation_loss:
                                f = open(f'{args.log_directory}/{args.model}_log.txt', "a")
                                f.write(f"Saving model on iteration {epoch}\n")
                                print(f"Saving model on iteration {epoch}\n")
                                torch.save(model, f"./models/{args.model}.pth")
                                f.close()
                                best_validation_loss = validation_loss

                model.train()
                
                # Save the losses
                training_losses.append(total_loss)
                validation_losses.append(validation_loss)

                # Save to log and print to output
                f = open(f'{args.log_directory}/{args.model}_log.txt', "a")
                f.write(f'iteration {epoch}s total loss: {total_loss}, validation loss: {validation_loss}\n')
                print(f'iteration {epoch}s total loss: {total_loss}, validation loss: {validation_loss}\n')
                f.close()

                # Save the losses
                np.save(f'{args.log_directory}/{args.model}_training_losses.npy', training_losses)
                np.save(f'{args.log_directory}/{args.model}_validation_losses.npy', validation_losses)


if __name__ == "__main__":
    main()