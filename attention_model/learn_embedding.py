import torch
import numpy as np
from tqdm import tqdm
from utils import initialize_dataloader, initialize_model, init_args


def evaluate(model, dataloader, criterion):
        loss = 0
        model.eval()
        with torch.no_grad():
                for observations, actions, target in  dataloader:
                        predictions = model.forward(observations, actions)
                        loss = criterion(predictions.flatten().float(), target.flatten().float())
                        loss += loss.item()
        model.train()
        return loss


def log(validation_loss, epoch, training_loss, args):
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


def main():

        args = init_args()

        # Open log file
        f = open(f'{args.model_dir}/log/log.txt', 'a')
        f.write("Starting training.\n")
        f.close()

        # Prepare the dataloader
        training_dataloader, validation_dataloader, test_dataloader, args = initialize_dataloader(args, subset=None)
        
        # Prepare the model
        model, criterion, optimizer = initialize_model(args)

        best_validation_loss = evaluate(model,validation_dataloader, criterion)
        training_losses = []
        validation_losses = []
        model.train()
        epoch = 0        

        # Iterate through the data
        while best_validation_loss > 1e-4:

                epoch += 1
                training_loss = 0

                for observations, actions, target in tqdm(training_dataloader):

                        # Zero out the gradient for this round of updates
                        optimizer.zero_grad()

                        # Conduct a forward pass of the transformer
                        predictions = model.forward(observations, actions)

                        # Compare the output of the model to the target
                        loss = criterion(predictions.flatten().float(), target.flatten().float())

                        # Update the model
                        loss.backward()
                        optimizer.step()

                        training_loss += loss.item()

                # Check against the validation dataset
                validation_loss = evaluate(model,validation_dataloader, criterion)

                # Save the losses
                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
                np.save(f'{args.model_dir}/log/training_losses.npy', training_losses)
                np.save(f'{args.model_dir}/log/validation_losses.npy', validation_losses)

                # Update the logs
                log(validation_loss, epoch, training_loss, args)

                # Save model
                if validation_loss < best_validation_loss:
                        save_model(model, args, epoch)
                        best_validation_loss = validation_loss

        # Apply to test dataset
        test_loss = evaluate(model, test_dataloader, criterion)
        print(f'Test loss: {test_loss}')


if __name__ == "__main__":
    main()