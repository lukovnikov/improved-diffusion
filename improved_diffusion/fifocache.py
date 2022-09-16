from typing import Dict

import torch


class FIFOCacheDataset(object):
    def __init__(self, minsize=10000, maxsize=50000):
        super(FIFOCacheDataset, self).__init__()
        self.minsize, self.maxsize = minsize, maxsize
        self.overwriteindex = 0
        self.actualsize = 0
        self.content = None

    def cache_ready(self):      # cache is ready when it has reached the minimum number of elements
        if self.content is None or len(self.content) < self.minsize:
            return False
        else:
            return True

    def push(self, batch:Dict[str,torch.Tensor]):  # input is a dictionary of tensors, each with batch dimension as first dimension
        if self.content is None:
            # initialize content cache to its max size
            self.content = {            }
            for k, v in batch.items():
                self.content[k] = torch.zeros(*((self.maxsize,)+v.shape[1:]), dtype=v.dtype, device=torch.device("cpu"))

        for k, v in batch.items():
            maxlenv = min(len(v), self.maxsize - self.overwriteindex)
            frm = self.overwriteindex
            to = self.overwriteindex+maxlenv
            self.content[k][frm:to] = v[:maxlenv]
        self.actualsize = min(self.actualsize + maxlenv, self.maxsize)
        self.overwriteindex = self.overwriteindex + maxlenv
        if self.overwriteindex >= self.maxsize:
            self.overwriteindex = 0
            assert self.actualsize == self.maxsize

    # dataset API
    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.content)


def _tst_fifocache():
    c = FIFOCacheDataset(10, 100)
    for i in range(20):
        x = {"a": torch.rand(60, 5), "b": torch.rand(60, 6)}
        c.push(x)


if __name__ == '__main__':
    _tst_fifocache()