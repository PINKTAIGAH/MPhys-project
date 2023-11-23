import torch
import utils 
import torch.nn as nn
import torch.optim as optim
import config
import torchvision as tv
from dataset import NuIDDataset 
from torch.utils.data import DataLoader
from training_utils_copy import *
import scipy.stats as stats
from sparceNetPID_copy import SparceNetPID
import medmnist
from medmnist import INFO, Evaluator
from torchvision import transforms

# Define discriminator and generator objects 
# model = SparceNetPID(
#     imageChannels = config.IMAGE_CHANNELS,
#     numClasses = config.N_CLASSES,
#     imageDimentions= config.INPUT_SIZE,
# )
model = tv.models.resnet18(pretrained=False)
input_size = model.fc.in_features
model.fc = torch.nn.Linear(input_size, out_features=config.N_CLASSES)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Define optimiser for both discriminator and generator
optimiser = optim.Adam(
    model.parameters(), lr=config.LEARNING_RATE, betas=config.MOMENTUM
)


# Loss function for model trianing 
criterion = torch.nn.CrossEntropyLoss().to(config.DEVICE)

# Load previously saved models and optimisers if True
if config.LOAD_MODEL:
    utils.load_checkpoint(
        config.CHECKPOINT_GEN_LOAD, model, optimiser, config.LEARNING_RATE,
    )

# Initialise training dataset and dataloader
# train_dataset = NuIDDataset(
#     "/home/giorgio/Desktop/train", 
#     name = "train",
#     maxVoxels=config.MAX_VOXELS, 
#     maxParticles=config.MAX_PARTICLE,
#     length=1000,
#     imageDims=config.IMAGE_SIZE,
#     nClasses=config.N_CLASSES,
# )
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=config.BATCH_SIZE,
#     shuffle=True,
# )

# # Initialise validation dataset and dataloader
# val_dataset = NuIDDataset(
#     "/home/giorgio/Desktop/test", 
#     name = "test",
#     maxVoxels=config.MAX_VOXELS, 
#     maxParticles=config.MAX_PARTICLE,
#     length=2000,
#     imageDims=config.IMAGE_SIZE,
#     nClasses=config.N_CLASSES,
# )
# val_loader = DataLoader(
#     train_dataset,
#     batch_size=config.BATCH_SIZE,
#     shuffle=True,
# )

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize(mean=[.5], std=[.5]),
    transforms.Pad(40)
])

dataset_flag = "bloodmnist"
download = True

# Get dataclass attribute, ie: the location of the object containing desired data 
info = INFO[dataset_flag]
DataClass = getattr(medmnist, info["python_class"])

# load the data (ie: member of torch.utils.data.DataClass)
train_dataset = DataClass(split="train", download=download, transform=data_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

val_dataset = DataClass(split="test", download=download, transform=data_transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)


# Initialise Gradscaler to allow for automatic mixed precission during training
model_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

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
    binning = np.linspace(0, config.N_CLASSES, config.N_CLASSES+1, dtype=int)

    for (images, labels) in dataloaders['val']:
        images = images.type(torch.float32).to(config.DEVICE)
        labels = labels.type(torch.LongTensor).to(config.DEVICE)
        output = model(images,)
        _, preds = torch.max(output, 1)
        H, *_ = stats.binned_statistic_2d(labels.to('cpu').numpy(), preds.to('cpu').numpy(), None, bins=[binning, binning], statistic='count')
        confusion += H
    return confusion

confusion = get_confusion_matrix()
print(f"{confusion}")
# class_name = {0: "nuMuCC", 1: "nuECC", 2: "NC",}
class_name = {INFO[dataset_flag]["label"]}
print("--- Class Accuracy")
for t in range(confusion.shape[0]):
    print(f"{class_name[t]:6}: {100*(confusion[t,t] / confusion[t].sum()):.1f}")
print()