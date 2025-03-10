import torch.nn as nn
from parameters import*

class loss(object):
    def __init__(self, param):
        super(loss, self).__init__()
        self.loss_fn=nn.CrossEntropyLoss()

    def compute(self, pred, target):
        loss=self.loss_fn(pred,target)
        return loss
