##There will be 3 parts to this code:
# 1. Pytorch Dataset
# 2. Pytorch Model
# 3. Pytorch Training loop

# This program will be used to classify playing cards by suit only due to the data stcture of the images and I am lazy
# to classify each individual card I would need to make a folder for each individual card and I am not doing that

import torch 
import torch.nn as nn #these have functions for nerual networks
import torch.optim as optim #define optimizers
from torch.utils.data import DataLoader, Dataset # Needed in step 1 

# torchvision is used for working with images
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

import timm # used for image classifications
import os



# first make a class
class PlayingCardsData(Dataset):

    def __init__(self, data_dir, transform=None): # tells what the class to do
        # data_dir is the directory where the data is stored
        # transform is the preprocessing that we want to apply to the data, so we are resizing the images to the same size effectivly doing nothing 

        # the data_dir here will assign class labels to all subfolders and images in the directory
        # the subfolders are consdiered classes labels and the images labels are just called labels
        self.data = ImageFolder(data_dir, transform=transform) 
    
    def __len__(self): # the data loader needs to know how many examples we have of playing cards
        return len(self.data)

    def __getitem__(self, idx): # Will return one item of the dataset based on whats in the index
        return self.data[idx] # returns the image and the label
    
    #this will get the class names from the image folder
    def classes(self):
        return self.data.classes   



data_dir = 'C:\\Users\\VO1D_H3ART\\Desktop\\Programming\\PyTorch stuff\\Images'

# this is a dictionary that will take k and v (Keys and values) within the ImageFolder
# the class_to_idx will return the class labels from the image folder
# .items() is a method used for dictionaries to return the key value pairs
# the key is the class label and the value is the index
index_to_name = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()} # this will return the class names from the image folder
print(index_to_name)



# this will resize the images to 128x128 pixels
# the reason we do this is because the model expects us to have consitent inputs
transform = transforms.Compose([
    transforms.Resize((128, 128)), # in order to use resize you need to use a tuple

    # A tensor is something that stores data in the range of [0,1] and are used to accelrate GPU stuff
    
    transforms.ToTensor() # this will convert the image to a tensor
]) 

RunModel = PlayingCardsData(data_dir, transform) # this will load the data from the training folder

print(len(RunModel)) # this will return the number of images in the dataset
print(RunModel[100]) # this will return the image and the label 

image, label = RunModel[100] # this will return the image and the label
print(RunModel[100])
print(image.shape) # this will return the shape of the image



# this will iterate through the dataset
for image, label, in RunModel:
    break


# the batch size is the number of images that will be loaded at a time
# the shuffle is used to shuffle the data so that the model does not learn the order of the data
dataloader = DataLoader(RunModel, batch_size=32, shuffle=True) # this will load the data into the dataloader

for image, label in dataloader:
    print(image.shape) # this will return the shape of the image: it looks like this: [32, 3, 128, 128] where 32 is the batch size, 3 is the number of channels, and 128x128 is the size of the image
    print(label) # this will return the labels of the images
    break


## Step 2: Pytorch Model
# this is the model that we will use to classify the playing cards

class classifier(nn.Module):
    def __init__(self, num_classes = 4):
        super(classifier, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True) # this will load the model
        self.feature = nn.Sequential(*list(self.model.children())[:-1]) # this will take the model and remove the last layer
        outputsize = 1280

        # make a classifier
        self.classifier = nn.Linear(outputsize, num_classes) # this will make a linear layer that will take the output size and the number of classes


    def forward(self, x):
        x = self.feature(x)
        output = self.classifier(x)
        print(output)
        return output

model = classifier(num_classes=4) # this will load the model

example_out = model(image) # this will return the output of the model
print(example_out)

# Step 3: Pytorch Training Loop
# this is the training loop that will train the model

# loss function

criterion = nn.CrossEntropyLoss() # this is the loss function that we will use to train the model
optimizer = optim.Adam(model.parameters(), lr=0.001) # this is the optimizer that we will use to train the model: lr is the learning rate and it is constant

print(criterion(example_out, label))

# this block is just loading in the images and then running them through the dataloader
train_folder =  'C:\\Users\\VO1D_H3ART\\Desktop\\Programming\\PyTorch stuff\\Images'
train_dataset = PlayingCardsData(train_folder, transform=transform) # this will load the data from the training folder
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # this will load the data into the dataloader

# and epoch is one pass through the dataset
num_epochs = 5
train_losses = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this will check if the GPU is available
print(device)
model.to(device) # this will put the model on the GPU



for epoch in range(num_epochs):
    model.train() # this will put the model in training mode
    running_loss = 0.0 # this will keep track of the loss

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) # this will put the images and labels on the GPU
        optimizer.zero_grad() # this will zero out the gradients
        outputs = model(images) # this will return the output of the model
        loss = criterion(outputs, labels) # this will return the loss
        loss.backward() # this will backpropagate the loss
        optimizer.step() # this will update the weights
        running_loss += loss.item()
    train_losses.append(running_loss/len(train_loader))

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')