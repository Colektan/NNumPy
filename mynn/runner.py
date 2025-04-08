import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    Train, evaluate, save, load the model.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.train_iter = []
        self.dev_iter = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            for iteration, data in enumerate(tqdm(train_set)):
                train_X = data["data"]
                train_y = data["label"]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)
                self.train_iter.append(epoch * len(train_set) / train_set.batch_size + iteration)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if (iteration + 1) % log_iters == 0:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)
                    self.dev_iter.append(epoch*len(train_set) / train_set.batch_size + iteration)

                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
        self.best_score = best_score

    def evaluate(self, data_set):
        loss = 0
        score = 0
        count = 0
        for d in data_set:
            X = d["data"]
            y = d["label"]
            logits = self.model(X)
            loss += len(y) * self.loss_fn(logits, y)
            score += len(y) * self.metric(logits, y)
            count += len(y)
        score /= count
        loss /= count
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)