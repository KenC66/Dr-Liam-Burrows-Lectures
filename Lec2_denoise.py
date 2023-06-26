# ## 5. Image Denoising with MNIST
# 
# - Problem setup: Suppose we have an observed noisy image $x$, and we wish to recover a clean version of the image, $u$. We model using additive gaussian noise: $x = u + \epsilon$ where $\epsilon$ is the noise.
# - Let us set the data up, by extracting the MNIST images and adding gaussian noise to them.

# In[ ]:

import torch, os
from skimage import io, transform
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import ToTensor


# In[ ]:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self,img_size,in_channels,out_channels):
        super(CNN, self).__init__()
        f = [10,20,30,20,10,out_channels]
# Input to conv1 will be image of shape [batch_size,1,img_size,img_size]
        self.conv1 = nn.Sequential(   
            nn.Conv2d(in_channels=in_channels,out_channels=f[0],kernel_size=(3,3),padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2,2)) 
        )
        self.conv2 = nn.Sequential( 
            nn.Conv2d(in_channels=f[0],out_channels=f[1],kernel_size=(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)) 
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=f[1],out_channels=f[2],kernel_size=(3,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=f[2],out_channels=f[2],kernel_size=(3,3),padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=f[2],out_channels=f[3],kernel_size=(3,3),padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=f[3],out_channels=f[4],kernel_size=(3,3),padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2))
        
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=f[4],out_channels=f[5],kernel_size=(1,1)),
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


# In[ ]:


# In[ ]:




# ## Loss Function:
# 
# - In our denoising setup we assume no label. Our dataset consists of only observed noisy images, i.e. $\mathcal{D} = \{ \mathbf{x}_i \in \mathbb{R}^{h \times w} \}_{k=1}^n$. This approach (not using labels) is known as **unsupervised learning**.
# 
# - We will implement a classic denoising model, which is composed of two terms:
# $$ \mathcal{L}(\theta) = \sum_{k=1}^n || \nabla f(\mathbf{x}_i;\theta) ||_2 + \frac{\lambda}{2} || f(\mathbf{x}_i; \theta) - \mathbf{x}_i ||_2^2 , $$
# - where $\lambda>0$ is a parameter which we hand tune according to the strength of noise. If we have a large $\lambda$, the second term is more dominiant and our network output will be matched more closely to the input (i.e. $f(\mathbf{x}_i ; \theta) \approx \mathbf{x}_i$). If $\lambda$ is small, the first term will be more dominant and more smoothing will occur.

# In[ ]:


def up_shift(f):
    g = torch.zeros_like(f)
    g[:-1, :] = f[1:, :]
    g[-1, :] = f[-1, :]
    return g

def down_shift(f):
    g = torch.zeros_like(f)
    g[1:, :] = f[:-1, :]
    g[0, :] = f[0, :]
    return g

def left_shift(f):
    g = torch.zeros_like(f)
    g[:, :-1] = f[:, 1:]
    g[:, -1] = f[:, -1]
    return g

def right_shift(f):
    g = torch.zeros_like(f)
    g[:, 1:] = f[:, :-1]
    g[:, 0] = f[:, 0]
    return g


def grad(f):
    f_x = (left_shift(f) - right_shift(f))/2
    f_y = (down_shift(f) - up_shift(f))/2
    
    return torch.sqrt(f_x**2 + f_y**2 + 1e-7)

class denoising_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, denoised, noisy, lambdaP):
        
        TV_term = grad(denoised)
        
        Fit_Term = (lambdaP/2)*(denoised-noisy)**2
        
        loss = TV_term + Fit_Term

        return loss.mean()


# In[ ]:


def train(num_epochs, model, loaders):
    
    model.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, images in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images, requires_grad=True).to(device)   # batch x. b_x is of shape [100,28,28]
            #print(b_x.shape)
            b_x = b_x.unsqueeze(1) # make dimensions [100,1,28,28], rather than [100,28,28]
            #print(b_x.shape)
            output = model(b_x.float())  

            #print(output.shape)
            loss = loss_func(output,b_x,lambdaP)
                        
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()                # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))        

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        ###comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        ###flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
       
        
if __name__ == '__main__':    
    
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
    
    train_imgs = train_data.data
    print(train_imgs.shape)
    train_imgs = train_imgs.numpy()
    # store the clean images before adding noise
    train_imgs0 = train_imgs/255.0

    test_imgs = test_data.data
    test_imgs = test_imgs.numpy()
    test_imgs0 = test_imgs/255.0

    sigma = 70/255.0 # noise level
    train_imgs = (train_imgs/255.0 + np.random.normal(0,sigma,train_imgs.shape)).clip(0,1)
    test_imgs = (test_imgs/255.0 + np.random.normal(0,sigma,test_imgs.shape)).clip(0,1)

    loaders = {
        'train' : torch.utils.data.DataLoader(train_imgs, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
        
        'test'  : torch.utils.data.DataLoader(test_imgs, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
    }
    loaders




    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(train_imgs0[0],cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(train_imgs[0],cmap='gray')
    plt.pause(4)



    model = CNN(img_size=28, in_channels=1, out_channels = 1).to(device)


    # In[ ]:


    learning_rate = 0.001

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)   
    optimizer


    # In[ ]:


    total_step = len(train_imgs)
    print(total_step)
    train_imgs.shape

    from torch.autograd import Variable
    num_epochs = 10
    lambdaP = 8
    loss_func = denoising_loss()    
    train(num_epochs, model, loaders)


    # In[ ]:


    with torch.no_grad():
        #Plot multiple
        figure = plt.figure(figsize=(15, 12))
        cols, rows =4,4
        for i in range(1, int((cols * rows)/2) + 1):
            sample_idx = torch.randint(len(test_imgs), size=(1,)).item()
            noisy, clean = test_imgs[sample_idx], test_imgs0[sample_idx]
            noisy = noisy[np.newaxis,np.newaxis,:,:]
            b_x = torch.from_numpy(noisy).float()
            b_x = b_x.to(device)
            output = model(b_x)
            output = output.cpu()

            figure.add_subplot(rows,cols,2*i - 1)
            plt.title('noisy input')
            plt.axis("off")
            plt.imshow(noisy.squeeze(),cmap="gray")

            figure.add_subplot(rows,cols,2*i)
            plt.title('net denoised')
            plt.axis("off")
            plt.imshow(output.squeeze(),cmap="gray")

        plt.pause(4)


# ## Homework
# 
# - Try training the segmentation problem again with DICE loss function.
# - The DICE coefficient is a measure of how well two sets coincide with one another, i.e. how well two segmentation results are similar. We use the output of our network, $u$, and the ground truth, $GT$, to measure performance.
# $$ DICE(u,GT) = \frac{ 2 \cdot | u \cdot GT |}{|u| + |GT|} $$
# <p align="center">
# <img src="dice.png" width="250" title="DICE." >
# </p> 
# 
# - A score of 1 is a perfect match, whereas a score of 0 is the opposite. We want a score of 1 in segmentation.
# - To use this in a loss function, the loss function looks like:
# $$ DICELOSS(u,GT) = 1 - DICE(u,GT) $$

# In[ ]:

    plt.show()