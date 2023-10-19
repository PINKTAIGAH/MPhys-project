### Import files in parent folder
import sys
sys.path.append("../")

# Import libraries
import torch
import numpy as np
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
from tqdm import tqdm
from torchvision.transforms import transforms
import medmnist
from medmnist import INFO, Evaluator
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

### Import custom files
import config
import utils
from resnetMedMnist import ResnetMedMnist

"""
Define Datasets
"""

# Get dataclass attribute, ie: the location of the object containing desired data 
info = INFO[config.DATASET_FLAG]        # Contains info on dataset     
DataClass = getattr(medmnist, info["python_class"])

# load the data (ie: member of torch.utils.data.DataClass)
train_dataset = DataClass(split="train", download=True)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

test_dataset = DataClass(split="test", download=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

"""
Define Objects
"""

model = ResnetMedMnist(
    config.IMAGE_CHANNELS,
    config.NUM_CLASSES,
    config.IMAGE_SIZE,
    listResiduals=[6, 8]
).to(config.DEVICE)
scaler = torch.cuda.amp.GradScaler()
criterion = nn.BCEWithLogitsLoss()
optimiser = optim.SGD(
    model.parameters(),
    lr=config.LEARNING_RATE, 
    momentum=config.MOMENTUM
)
tqdm_loader = tqdm(train_loader)

### Begin training loop

loss_data = []
for epoch in range(config.NUM_EPOCHS):
    model.train()
    running_loss = []
    for idx, (inputs, targets) in enumerate(tqdm_loader):
        inputs = inputs.to(torch.float32).to(config.DEVICE)
        targets = targets.to(torch.float32).to(config.DEVICE)
        # forward pass
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        running_loss.append(loss)

        # backwards pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        epoch_loss = sum(running_loss)/len(running_loss)
        loss_data.append(epoch_loss.to("cpu").item())
        print(f"Epoch: {epoch+1}\tloss: {epoch_loss.item()}")
    
### Begin evaluation of model

with torch.no_grad():
    # We want to get the accuracy
    running_accuracy = []
    running_predicted = []
    running_target = []

    for x, y in test_dataset:
        x = torch.from_numpy(x).to(torch.float32).to(config.DEVICE).unsqueeze(0)
        y = torch.from_numpy(y).to(torch.float32).to(config.DEVICE)
        
        y_predicted = model(x).sigmoid()
        y_predicted_class = (y_predicted.round())       # Round 0-1 floats to either 0 or 1 integers

        running_predicted.append(y_predicted_class.item())
        running_target.append(y.item())
        
        # Finding accuracy of results
        if y_predicted_class.item() == y.item():
            running_accuracy.append(y_predicted_class.item())
        accuracy = len(running_accuracy)/len(test_dataset)
    print(f"\n accuracy = {accuracy:.4}")

# Compute and plot confusion matrix
conf_matrix = confusion_matrix(running_target, running_predicted)
sns.heatmap(conf_matrix, annot=True)
print(f"True Positive accuracy: {conf_matrix[1,1]/conf_matrix[1].sum()}")
print(f"True negative accuracy: {conf_matrix[0,0]/conf_matrix[0].sum()}")

plt.show()