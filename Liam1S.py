from torchvision import datasets
from torchvision.transforms import ToTensor
import os, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import platform, psutil, time
from pynvml import *

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)


#Visualisation
# Plot one train data
import matplotlib.pyplot as plt
i=10000
plt.imshow(train_data.data[i], cmap='gray')
plt.title('%i' % train_data.targets[i])
#plt.show()

#Plot multiple
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
#plt.show()

from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}
#loaders

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.hidden1 = nn.Linear(28*28,512)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(512,10)
        self.softmax = nn.Softmax(dim=0)
        
# [batch_size, 28, 28]
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the input (batch_size, 28* 28)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.softmax(x)
        return x



def train1(num_epochs, model, loaders):
    
    model.train()
        
    # Train the model
    total_step = len(loaders['train'])
    time0 = time.time()
    print('\ttrain1 [Model 1] for %d epochs, %d sets of data  each with batch size = %d' % 
          (num_epochs,total_step,loaders['train'].batch_size))
    for epoch in range(num_epochs):
        print ('Epoch %2d/%d Loss =' % (epoch + 1, num_epochs),end= " ")  
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)#[0]    
            
            #print(b_x.shape)
            #print(output.shape)
            #print(b_y.shape)
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()                # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                       print(f' {loss.item():.3f}@{i+1:d}', end="")               
        timea = time.time() - time0; timen=timea/(epoch+1)*(num_epochs-1-epoch)    
        if (epoch % 50 == 0) or (epoch+1)==num_epochs:     
            print(f"/{total_step:d} {timen/60:.2f} mins Left")
        else:
            print(f"/{total_step:d} {timen/60:.2f} mins Left", end='\r')

def test(model, epos, my_id):
    model.eval()
    test_loss = 0
    correct = 0
    i = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            output = model(data)             #print(data.shape, output.shape, target.shape)1
            test_loss += F.nll_loss(output, target) #).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            if i==0:
                fig,ax=plt.subplots(1,2)
                ax[0].imshow( data[0,0,:,:] )
                ax[0].set_title('GT %d Pred %d' % (target[0],pred[0]))
                ax[1].imshow( data[1,0,:,:] )
                ax[1].set_title('GT %d Pred %d' % (target[1],pred[1]))                
            i += 1
        test_loss /= len(loaders['test'].dataset)
        per = 100. * correct / len(loaders['test'].dataset)
        print('\nModel {:d} Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          my_id,test_loss, correct, len(loaders['test'].dataset),   per))
        fig.suptitle('Model %d First two images in Batch 0 [%d epochs] Accu: %.2f' % (my_id, epos, per))
    plt.savefig('Pred_by_model_%d.jpg' % my_id)


class CNN(nn.Module):  # MODEL 2
    def __init__(self):
        super(CNN, self).__init__()
# Input to conv1 will be image of shape [batch_size,1,28,28] (height and width are 28 for this example)
        self.conv1 = nn.Sequential(   
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(3,3),padding=1), #output of this conv is of shape [BS,10,28,28]
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2,2)) #output of this is [BS,10,14,14]
        )
        self.conv2 = nn.Sequential( 
            nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),padding=1), #output of this is [BS,20,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)) # output of this is [BS,20,7,7]
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=30,kernel_size=(3,3),padding=1), #Output of this [BS,30,7,7]
            nn.ReLU(),
            nn.Conv2d(in_channels=30,out_channels=30,kernel_size=(3,3),padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=20,kernel_size=(3,3),padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(3,3),padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)) #[BS,10,28,28]
        
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=1,kernel_size=(1,1)),
            nn.Flatten(1),
            nn.ReLU(),  ##
            nn.Linear(28*28,10),
            nn.Sigmoid()            
        )
            

    def forward(self, x,plot=False):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
 
        if plot:
            print('Input shape', x.shape)
            print('After layer 1', x1.shape)
            print('After layer 2', x2.shape)
            print('After layer 3', x3.shape)
            print('After layer 4', x4.shape)
            print('After layer 5', x5.shape)
            print('After layer 6', x6.shape)
        return x6

def train2(num_epochs, model, loaders):
    
    model.train()
        
    # Train the model
    total_step = len(loaders['train'])
    time0 = time.time()
    print('\ttrain2  [Model 2] for %d epochs, %d sets of data  each with batch size = %d' % 
          (num_epochs,total_step,loaders['train'].batch_size))    
    for epoch in range(num_epochs):

        print ('Epoch %2d/%d Loss =' % (epoch + 1, num_epochs),end= " ")  
        for i, (images, labels) in enumerate(loaders['train']):
            # images is of size [batch_size, 28, 28]
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y

            output = model(b_x)#[0]               print(i, b_y.shape, output.shape)

            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()                # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print(f' {loss.item():.2f}@{i+1:d}', end="")   
        timea = time.time() - time0; timen=timea/(epoch+1)*(num_epochs-1-epoch)           
        if (epoch % 50 == 0) or (epoch+1)==num_epochs:     
            print(f"/{total_step:d} {timen/60:.2f} mins to go ")
        else:
            print(f"/{total_step:d} {timen/60:.2f} mins to go ", end='\r') 
                        

if __name__ == '__main__':
    num_epochs = 5
    for ID in range(1,3):
        print('info1:', train_data, end ="")
        print(' info2:', train_data.data.size(), end ="")
        print(' and info3: ', train_data.targets.size())
        start_time = time.time()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        s24=1014**2

        print('\nRunning:',__file__,' on', device, '=', platform.node())

        model1 = Net()  # Case 1: Full Connected (dense matrix)
        model2 = CNN()  # Case 2: Convolutional  (sparse matrix)
        
        
        # Which = input("Choose model to train (test) 1 for FCN or 2 for CNN : ")
        # ID = int(  Which )
    # Train:
        sfile = 'model%d_sav.h5' % ID
        
        if ID==1:
            loss_func = nn.NLLLoss()   
            learning_rate = 0.01
            optimizer = optim.Adam(model1.parameters(), lr = learning_rate)           
            model = model1
            if os.path.isfile( sfile ):
                torch.load( sfile, map_location=device ) ########### Previous trained results loaded
            print('\tPrevious trained results in [%s] loaded' % sfile)
            train1(num_epochs, model, loaders)
        else:
            model = model2
            learning_rate = 0.001
            optimizer = optim.Adam(model2.parameters(), lr = learning_rate)   
            loss_func = nn.CrossEntropyLoss()
            if os.path.isfile( sfile ):
                torch.load( sfile, map_location=device ) ########### Previous trained results loaded
            print('\tPrevious trained results in [%s] loaded' % sfile)
            train2(num_epochs, model, loaders)
        optimizer
        torch.save(model, sfile) ################### Current trained results saved
        print('\tINFO: saved = %s' % sfile)
        end_time = time.time()
        if use_cuda:
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0);    info = nvmlDeviceGetMemoryInfo(h)
            mem = f'Free {info.free/s24:.1f} MB (out of {info.total/s24:.1f} MB)'
        else:
            free = int(psutil.virtual_memory().total - psutil.virtual_memory().available)
            tot = int(psutil.virtual_memory().total)
            mem = f'Free {free/s24:.1f} MB (out of {tot/s24:.1f} MB)'

        print('\tMemory usage:',mem, ' and  Train Time used = %.2f seconds' % (end_time-start_time) )
        #print(model)
        # im,label = train_data[0]
        # im = im.unsqueeze(1)
        # print(im.shape)
        #  output = model2(im,plot=True)

        # test to see the prediction
        test(model, num_epochs,ID)
    plt.pause(2)
