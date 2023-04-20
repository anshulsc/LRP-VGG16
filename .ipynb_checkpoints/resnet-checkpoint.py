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
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet50.fc(x)
        return x

# Create an instance of the ResNet model
model = ResNetModel(num_classes=4)
model.to(device)


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
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 1  #epochs changed

# Iterate x epochs over the train data
for epoch in range(epochs):  
    for i, batch in enumerate(train_loader, 0):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        #print("inputs",inputs)
        optimizer.zero_grad()
        outputs = model(inputs)
        #print("Outputs",outputs)
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
#print("inputs for first batch",inputs)
labels = labels.numpy()
outputs = model(inputs).max(1).indices.detach().cpu().numpy()

#print("Outputs for first batch",outputs)
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


def get_linear_layer_indices(model):
    layer_indices = []         #to separate out the linear layers, their indices are stored in this empty list
    offset = 0
    for i, layer in enumerate(model.children()):
        if isinstance(layer, nn.Linear):
            layer_indices.append(i + offset)
        elif isinstance(layer, nn.Sequential):
            sub_indices = get_linear_layer_indices(layer)
            layer_indices += [i + offset + idx for idx in sub_indices]
            offset += len(list(layer.parameters()))
        else:
            offset += len(list(layer.parameters()))
    return layer_indices


def apply_lrp_on_resnet50(model, image):
    image = torch.unsqueeze(image, 0)
    # >>> Step 1: Extract layers
    layers = list(model.resnet50.children())[:-1]
    print("layers====",layers)
              
    linear_layer_indices = get_linear_layer_indices(model)

    # >>> Step 2: Propagate image through layers and store activations
    n_layers = len(layers)
    print("n_layers",n_layers)
    activations = [image] + [None] * n_layers # list of activations
    print("Activations", activations)
    for layer in range(n_layers):
        if layer in linear_layer_indices:
            if layer == 176:
                activations[layer] = activations[layer].reshape((1, 2048))
        activation = layers[layer].forward(activations[layer])
        if isinstance(layers[layer], torch.nn.modules.pooling.AdaptiveAvgPool2d):
            activation = torch.flatten(activation, start_dim=1) #flatten the high dimensional activations to 2D tensors.Flattening preserves the spatial relationships between features and ensures that the fully connected layer can operate on all the features simultaneously.
        activations[layer+1] = activation

    # >>> Step 3: Replace last layer with one-hot-encoding
    #The output of the network is a vector of probabilities for each category, and one-hot encoding is used to convert this vector into a binary vector that represents the predicted category. The element of the vector with the highest probability is set to 1, while all others are set to 0. 
    output_activation = activations[-1].detach().cpu().numpy()
    max_activation = output_activation.max()
    one_hot_output = [val if np.any(val == max_activation) else 0 for val in torch.from_numpy(outputs).detach().numpy()]



    activations[-1] = torch.tensor(one_hot_output, device=device)


    # >>> Step 4: Backpropagate relevance scores
    relevances = [None] * n_layers + [activations[-1]]
    print("Relevances", relevances )
    # Iterate over the layers in reverse order
    for layer in range(0, n_layers)[::-1]:
        current = layers[layer]     #layer is an integer val
        # Treat max pooling layers as avg pooling
        if isinstance(current, torch.nn.MaxPool2d):
            layers[layer] = torch.nn.AvgPool2d(2)
            current = layers[layer]
        if isinstance(current, torch.nn.Conv2d) or \
           isinstance(current, torch.nn.AvgPool2d) or\
           isinstance(current, torch.nn.Linear):
            activations[layer] = activations[layer].data.requires_grad_(True)
            if layer <= 16:       
                rho = lambda p: p + 0.25*p.clamp(min=0); 
                incr = lambda z: z+1e-9

            # LRP-alpha-gamma rule >> Favor positive contributions (activations) and amplify them by a factor of gamma
            if layer <= 16:
                #The alpha-beta variant of LRP has been shown to improve the interpretability of neural networks by providing more stable and consistent attribution scores
                #The alpha-beta variant of LRP has been shown to be effective in explaining the behavior of neural networks that use ReLU activation functions.
                alpha = 2; gamma = 0.5
                rho = lambda p: p + alpha*(p.clamp(min=0)**gamma); 
                incr = lambda z: z+1e-9

            # Middle layers, LRP-epsilon >> Remove some noise / Only most salient factors survive
            if 17 <= layer <= 30: 
                rho = lambda p: p;                       
                incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data

            # Upper Layers, LRP-0 >> Basic rule
            if layer >= 31:       
                rho = lambda p: p;                       
                incr = lambda z: z+1e-9

            # Transform weights of layer and execute forward pass
            z = incr(new_layer(layers[layer],rho).forward(activations[layer]))      
            # Element-wise division between relevance of the next layer and z
            print("zzzzzzzzzzzzzzzzzzzzzzzz",z.shape)
            print("==================",relevances[layer+1].shape)
            s = (relevances[layer+1]/z).data          #relevances[layer+1]/z = This is done to scale the relevances in the next layer so that their sum equals 1, which is a common technique used in relevance propagation algorithms.                           
            # Calculate the gradient and multiply it by the activation
            (z * s).sum().backward(); 
            #In this specific code snippet, the backward pass is
            #  being performed to compute the gradients of the 
            # output with respect to the layer activations. 
            # These gradients are then used to compute the relevance values for each layer,
            # which represent the contribution of each neuron in that layer to the final output of the network.
            c = activations[layer].grad       
            # Assign new relevance values           
            relevances[layer] = (activations[layer]*c).data   #The code you provided specifically updates the relevance scores for each layer by multiplying the activations of that layer by the gradients of the same layer with respect to the output.                        
        else:
            relevances[layer] = relevances[layer+1] #If the current layer is not the output layer, the relevance values are instead assigned to the relevance values of the next layer, which has already been computed using the same method.

    # >>> Potential Step 5: Apply different propagation rule for pixels
    return relevances[0]


# %%

# Calculate relevances for first image in this test batch
image_id = 2
print("$$$$$$$$$$$$$$$$...................$$$$$$$$$$$$$$$$$$",inputs[image_id])   #$$$$$$$$$$$$$$$$$$$$
image_relevances = apply_lrp_on_resnet50(model, inputs[image_id])
image_relevances = image_relevances.to_dense()
#to_dense() method in this code is a function provided by the Python package torch.sparse, which is used to convert a sparse matrix to a dense matrix.
 # used to convert a sparse matrix represented by the image_relevances object to a dense matrix. Dense matrices, on the other hand, store all the elements, including the zeros.
image_relevances.permute(0,2,3,1).squeeze().detach().cpu().numpy()
image_relevances = np.interp(image_relevances, (image_relevances.min(),
                                                image_relevances.max()), 
                                                (0, 1))      #this code is used to normalize the values of a NumPy array image_relevances to a range between 0 and 1.
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
    image_relevances.permute(0,2,3,1).squeeze().detach().cpu().numpy()
    plt.show()
else:
    print("This image is not classified correctly.")


# %%
