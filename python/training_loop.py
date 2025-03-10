
from parameters import*
import torch.optim as optim
from loss import*
import torch

class training_loop(object):

    def __init__(self, param, model):
        super(training_loop, self).__init__()
        self.param=param
        self.optimizer=optim.Adam(model.parameters(), lr=param.learning_rate)
        self.model=model
        self.loss=loss(param)

    def train_step(self, epoch, dataloader):
        self.model.train()
        for i, (img, label) in enumerate(dataloader) :
            #print(label)
            pred=self.model(img)
            #print(pred)
            loss=self.loss.compute(pred,label)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        print(f'Epoch [{epoch+1}/{self.param.num_epoch}], Loss: {loss.item():.4f}')


    def test_step(self, epoch, dataloader):
        self.model.eval()
        correct=0
        total=0
        with torch.no_grad():
            for img, labels in dataloader:
                pred=self.model(img)
                loss=self.loss.compute(pred,labels)

                max, maxid=torch.max(pred, 1)
                total+=labels.size(0)
                correct+=(maxid==labels).sum().item()

        accuracy=correct/total
        print(f'Accuracy: {accuracy}')



    def train(self, train_loader, test_loader):
        print("Training Started")
        for epoch in range (0, self.param.num_epoch):
            self.train_step(epoch, train_loader)
            self.test_step(epoch, test_loader)
