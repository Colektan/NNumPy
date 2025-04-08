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
exp_dir = "results"
eval_batch = 5
batch_size = 100
hidden_size = 2000
label_size = 10
lr = 0.01
scheduler_type = "MultiStepLR"
milestones = [6000, 9000]
gamma = 0.5
epoch = 100
weight_decay = [5e-3, 5e-3, 5e-3, 5e-3]
# weight_decay = None# size_list = [(3, 3, 1, 0), (10, 3, 1, 0), (40, 3, 1, 0)]6
current_time = datetime.datetime.now()
result_dir = os.path.join(exp_dir, f"exp_{current_time.strftime('%Y-%m-%d-%H-%M-%S')}")
os.mkdir(result_dir)
with open(os.path.join(result_dir, "params.json"), "w") as f:
    param_dict = {
        "eval_batch": eval_batch,
        "batch_size": batch_size,
        "hidden_size": hidden_size,

        # "size_list": size_list,
        "epoch": epoch,
        "learning rate" : lr,
        "scheduler": scheduler_type,
        "milestone": milestones,
        "gamma": gamma,
        "weight_decay": weight_decay,
        "eval_batch": eval_batch
    }
    json.dump(param_dict, f, indent=4)
    

# get data mean and build transformation
def rescale_transform(image):
    image = np.array(image)
    return image / 255.0
train_set = nn.dataset.CIFAR10(dir_path=data_dir, mode="train", eval_batch=eval_batch, transformation=rescale_transform)
train_loader = nn.dataloader.Dataloader(train_set, sampler=nn.sampler.SectionRandomSampler(train_set))
eval_set = nn.dataset.CIFAR10(dir_path=data_dir, mode="eval", eval_batch=eval_batch, transformation=rescale_transform)
eval_loader = nn.dataloader.Dataloader(eval_set)

def get_mean_and_std(dataset):
    data = []
    for d in dataset:
        data += d["data"].tolist()
    temp = np.array(data)
    mean = temp.mean()
    std = temp.std()
    return mean, std

train_mean, train_std = get_mean_and_std(train_set)
eval_mean, eval_std = get_mean_and_std(eval_set)
print("train_mean", train_mean, "train_std", train_std)
print("eval_mean", eval_mean, "eval_std", eval_std)

def rescale_and_centralize_transform_train(image):
    image = np.array(image)
    image = image / 255.0
    image = (image - train_mean) / train_std
    return image

def rescale_and_centralize_transform_eval(image):
    image = np.array(image)
    image = image / 255.0
    image = (image - train_mean) / train_std
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

trans_1 = nn.transformation.ImageRandomFlip()
trans_2 = nn.transformation.ImageRandomCrop()
trans_0 = nn.transformation.ImageRandomJitter()
train_set.transformation = [reshape, rescale_and_centralize_transform_train, trans_1, trans_2, flatten]
eval_set.transformation = [rescale_and_centralize_transform_eval]
train_loader = nn.dataloader.Dataloader(train_set, sampler=nn.sampler.SectionRandomSampler(train_set), batch_size=batch_size)
eval_loader = nn.dataloader.Dataloader(eval_set, batch_size=len(eval_set))

# check whether successfully centralized
# print("Checking centralization effectivness...")
# train_mean, train_std = get_mean_and_std(train_set)
# eval_mean, eval_std = get_mean_and_std(eval_set)
# print("Centralized train mean:", train_mean, "train std: ", train_std)
# print("Centralized eval mean:", eval_mean, "eval std: ", eval_std)

# normalize from [0, 255] to [0, 1]
image_input_shape = len(train_set[0]["data"].reshape(-1))
model = nn.models.Model_MLP([image_input_shape, hidden_size, label_size], 'ReLU', weight_decay)
# model = nn.models.Model_CNN(size_list=size_list, act_func='ReLU', classes=10, lambda_list=weight_decay)
optimizer = nn.optimizer.SGD(init_lr=lr, model=model) 
if scheduler_type == "MultiStepLR":
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)
else:
    scheduler = None
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=label_size)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
runner.train(train_loader, eval_loader, num_epochs=epoch, log_iters=100, save_dir=f'{result_dir}/best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)
_.savefig(os.path.join(result_dir, "Acc&loss.svg"))
_.savefig(os.path.join(result_dir, "Acc&loss.png"))
with open(os.path.join(result_dir, "loss_results.pkl"), "wb") as f:
    pickle.dump([runner.train_iter, runner.train_loss, runner.train_scores, runner.dev_iter, runner.dev_loss, runner.dev_scores], f)
plt.show()