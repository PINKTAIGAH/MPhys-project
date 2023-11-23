import torch
import utils 
import torch.nn as nn
import torch.optim as optim
import config
from dataset import NumberProtonDataset 
from torch.utils.data import DataLoader
from training_utils import *
import scipy.stats as stats
from sparceNetPID import SparceNetPID


# Define discriminator and generator objects 
model = SparceNetPID(
    imageChannels = config.IMAGE_CHANNELS,
    numClasses = config.N_CLASSES,
    imageDimentions= config.INPUT_SIZE,
)

# Define optimiser for both discriminator and generator
optimiser = optim.Adam(
    model.parameters(), lr=config.LEARNING_RATE, betas=config.MOMENTUM
)

# Loss function for model trianing 
criterion = torch.nn.CrossEntropyLoss()

# Load previously saved models and optimisers if True
if config.LOAD_MODEL:
    utils.load_checkpoint(
        config.CHECKPOINT_GEN_LOAD, model, optimiser, config.LEARNING_RATE,
    )

# Initialise training dataset and dataloader
train_dataset = NumberProtonDataset(
    "/home/giorgio/Desktop/train", 
    name = "train",
    maxVoxels=config.MAX_VOXELS, 
    maxParticles=config.MAX_PARTICLE,
    length=1000,
    imageDims=config.IMAGE_SIZE,
    nClasses=config.N_CLASSES,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
)

# Initialise Gradscaler to allow for automatic mixed precission during training
model_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

# Initialise validation dataset and dataloader
val_dataset = NumberProtonDataset(
    "/home/giorgio/Desktop/test", 
    name = "test",
    maxVoxels=config.MAX_VOXELS, 
    maxParticles=config.MAX_PARTICLE,
    length=2000,
    imageDims=config.IMAGE_SIZE,
    nClasses=config.N_CLASSES,
)
val_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
)

# define dictionary to hodl dataloaders
dataloaders = {"train":train_loader, "val":val_loader}
dataset_sizes = {"train":2000, "val":2000}

# Train the model
model, statistic = train_model(model, config.NUM_EPOCHS, criterion, optimiser, dataloaders,dataset_sizes, config.DEVICE)

# Save the model if configured to do so
if config.SAVE_MODEL:
    utils.save_checkpoint(model, optimiser,)

# Print some statistics
model = model.eval().to(config.DEVICE)
def get_confusion_matrix():
    global model, dataloaders
    confusion = np.zeros((config.N_CLASSES, config.N_CLASSES))
    binning = np.linspace(0, 3, 5, dtype=int)

    for (uPlane, vPlane, wPlane, labels) in dataloaders['val']:
        uPlane = uPlane.type(torch.float32).to(config.DEVICE)
        vPlane = vPlane.type(torch.float32).to(config.DEVICE)
        wPlane = wPlane.type(torch.float32).to(config.DEVICE)
        labels = labels.type(torch.LongTensor).to(config.DEVICE)
        output = model(uPlane, vPlane, wPlane, config.DEVICE)
        _, preds = torch.max(output, 1)
        H, *_ = stats.binned_statistic_2d(labels.to('cpu').numpy(), preds.to('cpu').numpy(), None, bins=[binning, binning], statistic='count')
        confusion += H
    return confusion

confusion = get_confusion_matrix()
print(f"{confusion}")
class_name = {0: "CCQE", 1: "CCRes", 2: "NCQE",}
print("--- Class Accuracy")
for t in range(confusion.shape[0]):
    print(f"{class_name[t]:6}: {100*(confusion[t,t] / confusion[t].sum()):.1f}")
print()