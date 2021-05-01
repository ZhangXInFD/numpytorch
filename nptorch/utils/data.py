import numpy as np


class Dataset:

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class Dataloader:

    def __init__(self, dataset: Dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batchsize = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.__reset()

    def __reset(self):
        m = self.dataset.__len__()
        index_list = list(range(m))
        if self.shuffle:
            np.random.shuffle(index_list)
        self.index_iter = []
        if self.drop_last:
            index_list = index_list[:(m // self.batchsize * self.batchsize)]
        for i in range(0, m, self.batchsize):
            self.index_iter.append(index_list[i:(i + self.batchsize)])
        self.index_iter = list(reversed(self.index_iter))

    def __next__(self):
        if not self.index_iter:
            self.__reset()
            raise StopIteration
        else:
            index = self.index_iter.pop()
            return self.dataset.__getitem__(index)

    def __iter__(self):
        return self

