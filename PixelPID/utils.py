import torch
import config

def save_checkpoint(model, optimiser, filename="my_checkpoint.pth.tar"):
    """
    Save dictionary of parameters of model and optimiser to specidied directory 
    in order to be loaded at a later time.

    Parameters
    ----------
    model: torch.nn.Module instance
        Neural network model to be saved

    optimiser: torch.optim instance
        Optimiser of model to be saved

    filename: string, optional
        Directory where model and optimiser will be saved
    """
    print("==> Saving checkpoint")
    # Dictionary constaining model and optimiser state parameters
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimiser.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimiser, lr):
    """
    Load previously saved model and optimisers by assingning saved dictionaries 
    containing state parameters to inputted model and optimiser.

    Parameters
    ----------
    checkpoint_file: string
        Directory of file containing state dictionaries of previously saved model
        and optimiser

    model: torch.nn.Module instance
        Neural network model where state dictionary will be loaded 

    optimiser: torch.optim instance
        Optimiser of model where state dictionary will be loaded 

    lr: torch.TensorFloat
        Value of learning rate that is currently being used to train model
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # Load saved state dictionaries 
    model.load_state_dict(checkpoint["state_dict"])
    optimiser.load_state_dict(checkpoint["optimizer"])

    # Assign current learning rate to the optimiser
    for param_group in optimiser.param_groups:
        param_group["lr"] = lr