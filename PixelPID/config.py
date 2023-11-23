import torch
import torchvision.transforms as transform

"""
Defining Hyperparameters
"""
IMAGE_SIZE = (1280, 2048)
# INPUT_SIZE = (1280, 2048)
INPUT_SIZE = (640, 1024)
IMAGE_CHANNELS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
MOMENTUM = (0.5, 0.999)
NUM_EPOCHS = 10
MAX_VOXELS = 2000
MAX_PARTICLE = 20
NUM_WORKERS = 2
N_CLASSES = 3
SAVE_MODEL = True
LOAD_MODEL = False


INPUT_TRANSFORMATION = transform.Compose([
    transform.Resize(INPUT_SIZE, antialias=False),
    # transform.Normalize(
    #     [0 for _ in range(IMAGE_CHANNELS)],   # generalise for multi channel
    #     [4 for _ in range(IMAGE_CHANNELS)],
    # ),
])