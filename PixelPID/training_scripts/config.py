import torch

"""
Defining Hyperparameters
"""
IMAGE_SIZE = (28, 28, 28)
NUM_CLASSES = 1
IMAGE_CHANNELS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
MOMENTUM = 0.9
NUM_EPOCHS = 1

DATASET_FLAG = "vesselmnist3d"