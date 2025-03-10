
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from training_loop import*
from  datasets import*
from  network  import*
from parameters import*

param=parameters()
model=NN(param)
trainer=training_loop(param,model)


transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

# Download and load training dataset
path='../../../data2/'
train_dataset = torchvision.datasets.MNIST(root=path, train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True)

# Download and load test dataset
test_dataset = torchvision.datasets.MNIST(root=path, train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False)

def train():

    #img, label =test_dataset.__getitem__(0)
    #print(label)
    # x=model(img)
    # print("Shape {img.shape}", x)
    # plt.imshow(img.permute(1,2,0).numpy())
    # plt.show()

    trainer.train(train_loader, test_loader)


def main():
    train()

if __name__=="__main__":
    main()
