from abc import abstractmethod
import numpy as np
from glob import glob
import os
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class Dataset():
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def get_idx(idx):
        pass

    @abstractmethod
    def __len__():
        pass


class CIFAR10(Dataset):
    def __init__(self, dir_path='../dataset/cifar-10-batches-py', use_batch_cache=True, mode="train", eval_batch=0, transformation=None):
        """
        dir_path: A path pointing to the directory of the dataset. Example: 'cifar-10-batches-py'
        """
        # meta-info
        meta_data = unpickle(os.path.join(dir_path, "batches.meta"))
        self.num_cases_per_batch = meta_data[b'num_cases_per_batch']
        self.labels = meta_data[b'label_names']
        self.num_vis = meta_data[b'num_vis']
        self.transformation = transformation

        if mode == "train":
            self.all_batch = glob(os.path.join(dir_path, "data_batch_*"))
            if eval_batch != 0:
                del self.all_batch[eval_batch - 1]
            self.count = 0
            for b in self.all_batch:
                data = unpickle(b)
                self.count += len(data[b"data"])
            print(f"Get {self.count} images for train.")

            self.batch_cache = unpickle(self.all_batch[0]) # 使用 batch cache，一次只在内存中放置一个数据块
            self.current_batch = 0
            if not use_batch_cache:
                raise NotImplementedError
        elif mode == "eval" and eval_batch != 0:
            self.batch_cache = unpickle(os.path.join(dir_path, f"data_batch_{eval_batch}"))
            self.current_batch = 0
            self.count = len(self.batch_cache[b"data"])
            print(f"Get {self.count} images for eval.")
        else:
            self.batch_cache = unpickle(os.path.join(dir_path, "test_batch"))
            self.current_batch = 0
            self.count = len(self.batch_cache[b"data"])
            print(f"Get {self.count} images for test.")

    def get_idx(self, idx):
        if idx >= self.count:
            raise IndexError("Index out of range")
        # if target datum not in cache, then load it.
        if int(idx / self.num_cases_per_batch) != self.current_batch:
            self.current_batch = int(idx / self.num_cases_per_batch)
            self.batch_cache = unpickle(self.all_batch[self.current_batch])

        idx -= self.current_batch * self.num_cases_per_batch
        datum = self.batch_cache[b"data"][idx]
        if self.transformation:
            if isinstance(self.transformation, list):
                for f in self.transformation:
                    datum = f(datum)
            elif callable(self.transformation):
                datum = self.transformation(datum)
            else:
                raise TypeError("Unsupported transformation")
        return {"data": datum, "label": self.batch_cache[b"labels"][idx]}
    
    def __getitem__(self, idx):
        return self.get_idx(idx)

    def __len__(self):
        return self.count



