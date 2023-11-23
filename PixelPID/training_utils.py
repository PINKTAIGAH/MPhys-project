import numpy as np, torch, torchvision as tv
import sklearn.metrics as skm

def train_model(model, epochs, criterion, optimizer, dataloaders, dataset_sizes, device, ):
    """Runs the training loop for a model

        Args:
            model: The model to be trained
            epochs: The number of epochs to train for
            criterion: The loss function to be optimised
            optimizer: The optimizer to use
            dataloaders: A dictionary containing the training ('train') and validation ('val') data loaders
            dataset_sizes: The number of samples in the training ('train') and validation ('val') datasets
            device: The device on which training should be run
            output_filename: The output model filename

        Returns:
            A tuple containing the best model and the metric history
    """
    # initialise a map to hold the training statistics generate throughout training
    statistics = {
        'train': { 'loss': [], 'accuracy': [], 'f1': [] },
        'val': { 'loss': [], 'accuracy': [], 'f1': [] }
    }

    epoch = 1
    learning = True
    while learning:
        for phase in ['train', 'val']:
            # TASK - Set the model to training or evaluation mode according to the current phase
            if phase == "train":
              model.train()
            else:
              model.eval()

            # Initialise variables used to calculate statistics for each epoch
            predictions = [None] * len(dataloaders[phase])
            truths = [None] * len(dataloaders[phase])
            running_loss = 0.0

            # iterate over the batches in the current dataloader
            for b, (uPlane, vPlane, wPlane, labels) in enumerate(dataloaders[phase]):
                # transfer the inputs and the labels to the GPU
                uPlane = uPlane.type(torch.float32).to(device)
                vPlane = vPlane.type(torch.float32).to(device)
                wPlane = wPlane.type(torch.float32).to(device)
                labels = labels.type(torch.LongTensor).to(device)
                model.to(device)

                optimizer.zero_grad()

                # operations on the model should compute and track gradients during training, but not during validation
                with torch.set_grad_enabled(phase == 'train'):
                    # TASK - run the network over the input images
                    outputs = model(uPlane, vPlane, wPlane, device)
                    loss = criterion(outputs, labels) # replace the placeholders with suitable values to compute the loss function
                    _, preds = torch.max(outputs, 1) # to which dimension should the network output be reduced? Think about the structure of outputs, what does it look like after each call?

                    # Should these steps always run?
                    if phase == "train": # chooce the appropriate phase in which to run these steps
                        loss.backward() # compute the loss with respect to each weight
                        optimizer.step() # update the value of each weight

                    predictions[b] = preds
                    truths[b] = labels.data

                # Keep track of the running loss, factoring batch size - the final batch can sometimes be smaller than
                # the specified batch size, so we need to account for that to ensure a correct loss for each epoch
                running_loss += loss.item() * uPlane.size(0)

            # calculate the loss, accuracy and f1 score for the epoch (stored in statistics)
            update_statistics(statistics[phase], running_loss / dataset_sizes[phase], predictions, truths)
            # print the statistics for the epoch
            print_statistics(statistics, phase, epoch)

        print()
        epoch += 1
        # Exit condition
        if epoch > epochs:
            learning = False


    return model, statistics

def update_statistics(phase_statistics, loss, predictions, truths):
    """Update the statistics dictionary

        Args:
            phase_statistics: The dictionary of statistics for a phase
            epoch: The epoch of training to which the statistics apply
            loss: The loss for the epoch
            predicitions: The network predictions for the epoch
            truths: The true classifications
    """
    preds_flat = torch.cat(predictions).cpu().numpy().flatten()
    truths_flat = torch.cat(truths).cpu().numpy().flatten()
    phase_statistics['loss'].append(loss)
    phase_statistics['accuracy'].append(skm.accuracy_score(truths_flat, preds_flat))
    phase_statistics['f1'].append(skm.f1_score(truths_flat, preds_flat, average='macro'))


def print_statistics(statistics, phase, epoch):
    """Print the statistics for a given phase and epoch

        Args:
            statistics: The dictionary of statistics
            phase: The phase of interest
            epoch: The epoch of training to which the statistics apply
    """
    print(f"=== {phase.title()} {epoch} ===")
    print('    Loss: {:.4f}'.format(statistics[phase]['loss'][-1]))
    print('    Accuracy: {:.4f}'.format(statistics[phase]['accuracy'][-1]))
    print('    F1: {:.4f}'.format(statistics[phase]['f1'][-1]))
