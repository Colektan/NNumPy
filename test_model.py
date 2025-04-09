import mynn as nn
from draw_tools.plot import plot
import datetime

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os
from glob import glob
import json

# fixed seed for experiment
np.random.seed(309)

data_dir = "dataset/cifar-10-batches-py/"
model_path = r"results\exp_2025-04-09-00-10-56\best_models\best_model.pickle"
    
# get data mean and build transformation
def rescale_and_centralize_transform_test(image):
    image = np.array(image)
    image = image / 255.0
    image = (image - 0.47324696247574394) / 0.25173213355349583
    return image

def reshape(image):
    '''
    in: 3072, out [3, 32, 32]
    '''
    image = image.reshape(3, 32, 32)
    return image

def flatten(image):
    '''
    in: [3, 32, 32], out [3, 32, 32]
    '''
    image = image.reshape(-1)
    return image

test_set = nn.dataset.CIFAR10(dir_path=data_dir, mode="test", transformation=rescale_and_centralize_transform_test)
test_loader = nn.dataloader.Dataloader(test_set, batch_size=len(test_set))

model = nn.models.Model_MLP()
model.load_model(model_path)

score = 0
count = 0
for d in test_loader:
        X = d["data"]
        y = d["label"]
        logits = model(X)
        score += len(y) * nn.metric.accuracy(logits, y)
        count += len(y)
score /= count

print("Score:", score)