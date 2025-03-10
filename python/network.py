
import torch.nn as nn

class  NN(nn.Module):
    def __init__(self, param):
        super(NN, self).__init__()
        self.param=param
        self.h=param.height
        self.w=param.width
        self.c=param.channels
        self.n_classes=param.n_classes

        self.Linear1=nn.Linear(self.c*self.h*self.w, 512)
        self.act=nn.ReLU()
        self.Linear2=nn.Linear(512, 256)
        self.Linear3=nn.Linear(256, self.n_classes)


    def forward(self,x):
        x=x.view(-1,self.h*self.w*self.c)
        x=self.act(x)
        x=self.Linear1(x)
        x=self.act(x)
        x=self.Linear2(x)
        x=self.act(x)
        x=self.Linear3(x)
        return x



class  CNN(nn.Module):
    def __init__(self, input_h, input_w, input_c):
        super( CNN, self).__init__()
        self.h=input_h
        self.w=input_w
        self.c=input_c


    def foward(self):
        pass
