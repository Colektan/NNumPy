from abc import abstractmethod
import numpy as np

class Dataloader:
    def __init__(self, dataset, sampler=None, batch_size=1, post_process=None):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        if self.sampler:
            self.length = len(sampler)
        else:
            # sequential sampler
            self.current_idx = 0
            self.length = len(dataset)
        self.post_process = post_process
        self.stop = False

    def __iter__(self):
        self.stop = False
        if self.sampler:
            self.sampler.re_init_idx()
        else:
            self.current_idx = 0
        return self
    
    def allocate(self, data):
        assert type(data) == type([])
        keys = data[0].keys()
        new_d = {}
        for key in keys:
            new_d[key] = []
        for d in data:
            for key in keys:
                new_d[key].append(d[key])
            
        for k,v in new_d.items():
            new_d[k] = np.array(v)
        
        return new_d

    def __next__(self):
        if self.stop:
            raise StopIteration
        if self.sampler:
            data = []
            for num in range(self.batch_size):
                idx = self.sampler.get_idx()
                if not idx is None:
                    data.append(self.dataset[idx])
                else:
                    self.stop = True
        else:
            data = []
            for num in range(self.batch_size):
                if self.current_idx < self.length:
                    data.append(self.dataset[self.current_idx])
                    self.current_idx += 1
                else:
                    self.stop = True
        if len(data) == 0:
            raise StopIteration
        data = self.allocate(data)
        if self.post_process:
            data = self.post_process(data)
        return data

    def __len__(self):
        return self.length
                    
