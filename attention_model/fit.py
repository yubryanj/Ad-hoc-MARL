import torch
import numpy as np
from tqdm import tqdm
from utils import initialize_dataloader, initialize_model, init_args, evaluate, save_model, log


def main():

        args = init_args()

        # Prepare the dataloader
        train_dataloaders, validation_dataloaders, test_dataloader, args = initialize_dataloader(args, subset=None)
        
        # Prepare the model
        model, criterion, optimizer = initialize_model(args)

        best_validation_loss = evaluate(model,validation_dataloaders, criterion)
        log(-1, args, validation_loss = best_validation_loss)

        training_losses = []
        validation_losses = []
        model.train()
        epoch = 0        

        # Iterate through the data
        while best_validation_loss > args.threshold:

                epoch += 1.0
                training_loss = 0.0
                n_batches = 0.0

                env = np.random.choice(args.training_agents)

                for observations, actions, target in tqdm(train_dataloaders[env]):

                        # Zero out the gradient for this round of updates
                        optimizer.zero_grad()

                        # Conduct a forward pass of the transformer
                        prediction = model.forward(observations, actions)

                        # Compare the output of the model to the target
                        prediction = torch.Tensor(prediction.flatten())
                        target = torch.Tensor(target.flatten().float())
                        loss = criterion(prediction, target)

                        # Update the model
                        loss.backward()
                        optimizer.step()

                        training_loss += loss.item()
                        n_batches += 1.0

                # Check against the validation dataset
                validation_loss = evaluate(model, validation_dataloaders, criterion)

                # Scale by the batch size
                training_loss = training_loss / n_batches

                # Save the losses
                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
                np.save(f'{args.model_dir}/log/training_losses.npy', training_losses)
                np.save(f'{args.model_dir}/log/validation_losses.npy', validation_losses)

                # Update the logs
                log(epoch, args, validation_loss = validation_loss, training_loss=training_loss)

                # Save model
                if validation_loss < best_validation_loss:
                        save_model(model, args, epoch)
                        best_validation_loss = validation_loss

        # Apply to test dataset
        test_loss = evaluate(model, test_dataloader, criterion)
        log(-2, args, test_loss = test_loss)


if __name__ == "__main__":
    main()