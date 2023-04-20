# %% Imports
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models import resnet50
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import copy
import pandas as pd 
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:54.1"
# Set GPU device
print(torch.cuda.is_available())
device = torch.device("cuda:0")


# %% Load data
TRAIN_ROOT = "D:/research/xai-series-master/xai-series-master/data/brain_mri/training"
TEST_ROOT = "D:/research/xai-series-master/xai-series-master/data/brain_mri/testing"
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_ROOT)
test_dataset = torchvision.datasets.ImageFolder(root=TEST_ROOT)


# %% Building the model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.resnet50 = resnet50(pretrained=True)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, 4)
    
    def forward(self, x):
        x = self.resnet50(x)
        return x

model = CNNModel()
model.to(device)
model

# %% Prepare data for pretrained model
train_dataset = torchvision.datasets.ImageFolder(
        root=TRAIN_ROOT,
        transform=transforms.Compose([
                      transforms.Resize((255,255)),
                      transforms.ToTensor()
        ])
)

test_dataset = torchvision.datasets.ImageFolder(
        root=TEST_ROOT,
        transform=transforms.Compose([
                      transforms.Resize((255,255)),
                      transforms.ToTensor()
        ])
)

#train_dataset[0][0].permute(1,2,0)

# %% Create data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

# %% Train
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
epochs = 1 #epochs changed

# Iterate x epochs over the train data
for epoch in range(epochs):  
    for i, batch in enumerate(train_loader, 0):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        #print("Outputs here",outputs) #*******************
        # Labels are automatically one-hot-encoded
        loss = cross_entropy_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        print("This is loss-->",loss)

# %% Inspect predictions for first batch
import pandas as pd
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.numpy()
outputs = model(inputs).max(1).indices.detach().cpu().numpy()
comparison = pd.DataFrame()
print("Batch accuracy: ", (labels==outputs).sum()/len(labels))
comparison["labels"] = labels

comparison["outputs"] = outputs
comparison

# %% Layerwise relevance propagation for VGG16
# For other CNN architectures this code might become more complex
# Source: https://git.tu-berlin.de/gmontavon/lrp-tutorial
# http://iphome.hhi.de/samek/pdf/MonXAI19.pdf

def new_layer(layer, g):
    """Clone a layer and pass its parameters through the function g."""
    layer = copy.deepcopy(layer)
    try: layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError: pass
    try: layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError: pass
    return layer

def dense_to_conv(layers):
    """ Converts a dense layer to a conv layer """
    newlayers = []
    for i,layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            newlayer = None
            if i == 0:
                m, n = 512, layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,7,7))
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))
            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers

def get_linear_layer_indices(model):
    offset = len(model.resnet50.layer1) + len(model.resnet50.layer2) + len(model.resnet50.layer3) + 2    #$$$$$$$$$$$$$
    indices = []
    for i, layer in enumerate(model.resnet50.layer4): #$$$$$$$$$$$$$$$$$$$$
        if isinstance(layer, nn.Linear):
            indices.append(i)
    indices = [offset + val for val in indices]
    return indices

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

def apply_lrp_on_resnet50(model, image):
    image = torch.unsqueeze(image, 0)
    # >>> Step 1: Extract layers
    layers = list(model.children())[:-1] # remove last FC layer
    linear_layer_indices = get_linear_layer_indices(model)
    # >>> Step 2: Propagate image through layers and store activations
    n_layers = len(layers)
    activations = [image] + [None] * n_layers # list of activations
    
    for layer in range(n_layers):
        if layer in linear_layer_indices:
            if layer == n_layers-1:
                activations[layer] = activations[layer].reshape((1, -1))
        activation = layers[layer].forward(activations[layer])
        if isinstance(layers[layer], nn.AdaptiveAvgPool2d):
            activation = torch.flatten(activation, start_dim=1)
        activations[layer+1] = activation

    # >>> Step 3: Replace last layer with one-hot-encoding
    output_activation = activations[-1].detach().cpu().numpy()
    max_activation = output_activation.max()
    one_hot_output = np.where(output_activation == max_activation, output_activation, 0)

    activations[-1] = torch.tensor(one_hot_output, device=device)




    # >>> Step 4: Backpropagate relevance scores
    relevances = [None] * n_layers + [activations[-1]]
    # Iterate over the layers in reverse order
    for layer in range(0, n_layers)[::-1]:
        current = layers[layer]     #layer is an integer val
        # Treat max pooling layers as avg pooling
        if isinstance(current, nn.MaxPool2d):
            layers[layer] = nn.AvgPool2d(2)
            current = layers[layer]
        if isinstance(current, nn.Conv2d) or \
           isinstance(current, nn.AvgPool2d) or\
           isinstance(current, nn.Linear):
            activations[layer] = activations[layer].data.requires_grad_(True)
            
            # Apply variants of LRP depending on the depth
            # see: https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
            # Lower layers, LRP-gamma >> Favor positive contributions (activations)
            if layer <= 16:       
                rho = lambda p: p + 0.25*p.clamp(min=0); 
                incr = lambda z: z+1e-9+0.5*((z**2).mean()**.5).data
            # Middle layers, LRP-epsilon >> Remove some noise / Only most salient factors survive
            if 17 <= layer <= 30: 
                rho = lambda p: p;                       
                incr = lambda z: z+1e-9+0.5*((z**2).mean()**.5).data
            # Upper Layers, LRP-0 >> Basic rule
            if layer >= 31:       
                rho = lambda p: p;                       
                incr = lambda z: z+1e-9
                
            # Apply alpha beta variant of LRP
            # see: https://doi.org/10.1007/978-3-030-28954-6_10
            if layer <= 16:
                alpha = 2; beta = 1
            else:
                alpha = 1; beta = 0
            
            # Transform weights of layer and execute forward pass
            z = incr(new_layer(layers[layer],rho).forward(activations[layer]))
            # Element-wise division between relevance of the next layer and z
            s = (relevances[layer+1]/z).data                                     
            # Calculate the gradient and multiply it by the activation
            (z * s).sum().backward(); 
            #In this specific code snippet,
            c = activations[layer].grad 

            #In this specific code snippet, the backward pass is
            #  being performed to compute the gradients of the 
            # output with respect to the layer activations. 
            # These gradients are then used to compute the relevance values for each layer,
            relevances[layer] = (activations[layer]*c).data
        else:
            relevances[layer] = relevances[layer+1]
    # >>> Potential Step 5: Apply different propagation rule for pixels
    return relevances[0]
   
            #relevances[0] = relevances[0].sum(dim=1, keepdim=True)
    #relevance_per_pixel = relevances[0] / torch.sum(relevances[0])
    #image_relevance = torch.mul(image, relevance_per_pixel).sum(dim=1, keepdim=True)
    #return image_relevance


# %%

# Calculate relevances for first image in this test batch
image_id = 24
print("$$$$$$$$$$$$$$$$",inputs[image_id])
image_relevances = apply_lrp_on_resnet50(model, inputs[image_id])
image_relevances = image_relevances.permute(0,2,3,1).detach().cpu().numpy()[0]
image_relevances = np.interp(image_relevances, (image_relevances.min(),
                                                image_relevances.max()), 
                                                (0, 1))
# Show relevances
pred_label = list(test_dataset.class_to_idx.keys())[           #pred_label is a variable that stores the predicted label for a given image based on the class index of the test dataset
             list(test_dataset.class_to_idx.values())
            .index(labels[image_id])]
print("length of image id ---> ",len(outputs))
print("length of lable ---> ",len(labels))
print("---------- ---> ",outputs[image_id])
print("//////////////// ---> ",labels[image_id])
#outputs and labels are lists that store the model's predicted output and ground truth label for each image in the test dataset.
if outputs[image_id] == labels[image_id]:         #image_id is an index that represents the current image being evaluated
    print("Groundtruth for this image: ", pred_label)

    # Plot images next to each other
    plt.axis('off')
    plt.subplot(1,2,1)
    #If the model's prediction is correct, it prints the predicted label for the image, and then displays a plot of the image's relevance heatmap (image_relevances) and the image itself.
    #gray_image = np.mean(image_relevances, axis=0)
    #plt.imshow(gray_image, cmap='seismic')
    plt.imshow(image_relevances[:,:,0], cmap="seismic")
    plt.subplot(1,2,2)
    plt.imshow(inputs[image_id].permute(1,2,0).detach().cpu().numpy())
    plt.show()
else:
    print("This image is not classified correctly.")


# %%
