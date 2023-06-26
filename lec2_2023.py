#!/usr/bin/env python     [>jupyter nbconvert --to script lec2_2023.ipynb]
# coding: utf-8

# In[1]:


import torch, os
from skimage import io, transform
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# ## UNET
# - UNet is a particular type of CNN, named after its U shape
# - [Original paper](https://arxiv.org/pdf/1505.04597.pdf)
# <p align="center">
# <img src="unet.png" width="1000" title="Image." >
# </p>

# In[2]:


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super().__init__()

        self.activation_fn = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation_fn(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation_fn(x)
        return x
    
class down_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x,p

class up_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0)
        self.conv = conv_block(out_channels+out_channels, out_channels)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels = 1, f = [64,128,256,512,1024]):
        super().__init__()

        self.encoder1 = down_block(in_channels, f[0])
        self.encoder2 = down_block(f[0],f[1])
        self.encoder3 = down_block(f[1],f[2])
        self.encoder4 = down_block(f[2],f[3])

        self.bottleneck = conv_block(f[3],f[4])

        self.decoder1 = up_block(f[4],f[3])
        self.decoder2 = up_block(f[3],f[2])
        self.decoder3 = up_block(f[2],f[1])
        self.decoder4 = up_block(f[1],f[0])

        self.outputs = nn.Conv2d(f[0], out_channels, kernel_size = 1, padding = 0)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, inputs):
        c1, p1 = self.encoder1(inputs)
        c2, p2 = self.encoder2(p1)
        c3, p3 = self.encoder3(p2)
        c4, p4 = self.encoder4(p3)

        bn = self.bottleneck(p4)

        d1 = self.decoder1(bn, c4)
        d2 = self.decoder2(d1, c3)
        d3 = self.decoder3(d2, c2)
        d4 = self.decoder4(d3, c1)

        outputs = self.outputs(d4)
        outputs = self.sigmoid_layer(outputs)
        
        plot = True
        if plot:
            print('Input shape', inputs.shape)
            print('After layer 1', p1.shape)
            print('After layer 2', p2.shape)
            print('After layer 3', p3.shape)
            print('After layer 4', p4.shape)
            print('After bn', bn.shape)
            print('After layer 6', d1.shape)
            print('After layer 7', d2.shape)
            print('After layer 8', d3.shape)
            print('After layer 9', d4.shape)
            print('Output shape', outputs.shape)

        return outputs
UNet = UNet()

x = torch.zeros([1,3,256,256])
y = UNet(x)


# ## Segmentation
# 
# - Image segmentation is the task of partitioning an image, or identifying an object in an image
# - Particular value in medical imaging, highlighting objects of interest
# 
# <p align="center">
# <img src="DRIVE/training/images/21_training.png" width="200" title="Image." >
# </p>
# <p align="center">
# <img src="DRIVE/training/1st_manual/21_manual1.png" width="200" title="Image." >
# </p>
# 
# - The target image is a binary image, where pixels = 0 are background and pixels = 1 are foreground. We want the network to output a binary image replicating this.

# In[3]:


def get_paths(base_path='DRIVE/'):
    train_im_paths = []
    train_gt_paths = []
    test_im_paths = []
    test_gt_paths = []
    
    for i in range(21, 41):
        train_im_paths.append(base_path + 'training/images/%d_training.tif'%(i))
        train_gt_paths.append(base_path + 'training/1st_manual/%d_manual1.gif'%(i))

    for i in range(1, 21):
        test_im_paths.append(base_path + 'test/images/%d_test.tif'%(i))
        test_gt_paths.append(base_path + 'test/1st_manual/%d_manual1.gif'%(i))
        
    train_paths = [train_im_paths,train_gt_paths]
    test_paths = [test_im_paths,test_gt_paths]
    return train_paths, test_paths

def read_and_resize(im_paths, gt_paths, resize=(256, 256, 1)): 
    imgs = io.imread(im_paths, 1)
    gts = io.imread(gt_paths, 1) 

    imgs = transform.resize(imgs, resize)
    gts = transform.resize(gts, resize) 
        
    return imgs, gts


# In[4]:


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        self.im_paths = paths[0]
        self.gt_paths = paths[1]
        self.preprocesses = T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
        ])
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        image = Image.open(self.im_paths[index])
        label = Image.open(self.gt_paths[index])
        # 2. Preprocess the data (e.g. torchvision.Transform).
        
        image = self.preprocesses(image)
        label = self.preprocesses(label)
        
        #Ensure gt is binary
        label[label>.5] = 1
        label[label<=.5]=0
        # 3. Return a data pair (e.g. image and label).
        return image, label
        
    def __len__(self):
        return len(self.im_paths)
    




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



def train(num_epochs, model, train_loader):
    
    model.train()
        
    # Train the model
    total_step = len(train_loader)
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)  
            #loss = loss_func(torch.squeeze(output), torch.squeeze(b_y))
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()                # apply gradients             
            optimizer.step()                
            
            if (i+1) % 2 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))               

 
if __name__ == '__main__':  
    if os.path.dir('DRIVE/') :
        train_paths, test_paths = get_paths()
        print('Image data from DIRVE/  read ...')
    else:
        print("Error as DIR = DIRVE is not there ..."); exit()
    custom_dataset = CustomDataset(train_paths)
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                            batch_size=4, 
                                            shuffle=True)

    # In[5]:

    print(train_loader)
    data_iter = iter(train_loader)
    image,label = next( data_iter )
    print(image.size())
    print(label.size())

    label = torch.squeeze(label)
    print(label.size())


    # In[ ]:

    im,label = custom_dataset[0]
    print(im.size())
    print(label.size())

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im.permute(1, 2, 0))
    plt.subplot(1,2,2)
    plt.imshow(label[0,:,:])


    # In[ ]:
    model = CNN(img_size=256,in_channels=3, out_channels=1)


    # In[ ]:


    learning_rate = 0.001

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)   
    loss_func = nn.BCELoss()   


    # In[ ]:


    from torch.autograd import Variable

    num_epochs = 200   
        
    train(num_epochs, model, train_loader)


    # In[ ]:


    test_dataset = CustomDataset(test_paths)
    train_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=4, 
                                            shuffle=True)
    dataset = test_dataset

    with torch.no_grad():
        figure = plt.figure(figsize=(10, 8))
        rows = 2
        for i in range(0, rows):
            sample_idx = torch.randint(len(dataset), size=(1,)).item()
            img, label = dataset[sample_idx]
            b_x = img.unsqueeze(0)
            output = model(b_x)

            figure.add_subplot(rows, 3, 3*i + 1)
            plt.axis("off")
            plt.imshow(img.permute(1,2,0))
            plt.title("image")
            figure.add_subplot(rows, 3, 3*i + 2)
            plt.axis("off")
            plt.imshow(torch.squeeze(output), cmap="gray")
            plt.title("model output")
            figure.add_subplot(rows, 3, 3*i + 3)
            plt.axis("off")
            plt.imshow(torch.squeeze(label), cmap="gray")
            plt.title("ground truth")
        plt.show()


