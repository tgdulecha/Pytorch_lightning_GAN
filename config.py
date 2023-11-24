import os
import torch

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
MAX_EPOCHS = 5

# Dataset
DATA_DIR = "dataset/"
# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16