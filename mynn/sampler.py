from abc import abstractmethod
import numpy as np

class Sampler:
    """
    Modify your own 
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_idx(self):
        pass

    @abstractmethod
    def re_init_idx(self):
        pass


class RandomSampler(Sampler):
    def __init__(self, dataset):
        self.length = len(dataset)
        self.idxes = np.arange(self.length)
        np.random.shuffle(self.idxes)
        self.current = -1

    def get_idx(self):
        self.current += 1
        if self.current >= self.length:
            return None
        else:
            return self.idxes[self.current]
    
    def re_init_idx(self):
        self.current = -1
    
    def __len__(self):
        return self.length
    
class SectionRandomSampler(Sampler):
    """
    Random Samples in each section(10000 samples).
    """
    def __init__(self, dataset, section_size=10000):
        self.length = len(dataset)
        self.max_sections = int(self.length / section_size) + 1
        self.section_size = section_size
        self.random_section = np.arange(self.max_sections)
        np.random.shuffle(self.random_section)
        self.idxes = []
        for i in range(self.max_sections):
            target_section = self.random_section[i]
            if target_section == self.max_sections-1:
                temp = np.arange(self.length - section_size*target_section) + section_size*target_section
            else:
                temp = np.arange(section_size) + section_size*target_section
            np.random.shuffle(temp)
            self.idxes.append(temp)
        self.idxes = np.concatenate(self.idxes)
        self.current = -1

    def get_idx(self):
        self.current += 1
        if self.current >= self.length:
            return None
        else:
            return self.idxes[self.current]
    
    def re_init_idx(self):
        section_size = self.section_size
        self.random_section = np.arange(self.max_sections)
        np.random.shuffle(self.random_section)
        self.idxes = []
        for i in range(self.max_sections):
            target_section = self.random_section[i]
            if target_section == self.max_sections-1:
                temp = np.arange(self.length - section_size*target_section) + section_size*target_section
            else:
                temp = np.arange(section_size) + section_size*target_section
            np.random.shuffle(temp)
            self.idxes.append(temp)
        self.idxes = np.concatenate(self.idxes)
        self.current = -1
        self.current = -1
    
    def __len__(self):
        return self.length