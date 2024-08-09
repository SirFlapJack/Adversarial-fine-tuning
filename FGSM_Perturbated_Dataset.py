#Perturbed dataset

import os
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Load the pretrained ResNet-18 model
net = models.resnet18(pretrained=False, num_classes=10)
net.load_state_dict(torch.load('/content/image_recognition_model.pth', map_location=torch.device('cpu')))
net.eval()  # Ensure the model is in evaluation mode

def fgsm_attack(data, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

def generate_fgsm_adversarial_example(model, data, target, epsilon):
    data_copy = data.clone().detach().requires_grad_(True)  # Clone and set requires_grad to True
    output = model(data_copy)
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data_copy.grad.data
    perturbed_data = fgsm_attack(data_copy, epsilon, data_grad)
    return perturbed_data

# Data transformation and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# Parameters
epsilon = 0.01  # Start with zero perturbation to test
num_images =500   # Number of images to test with perturbation

# Create directory to save images
output_dir = 'perturbed_images'
os.makedirs(output_dir, exist_ok=True)

# Lists to store perturbed images and their labels
perturbed_images = []
perturbed_labels = []

for i, (data, target) in enumerate(test_loader):
    if i >= num_images:
        break

    # Move data and target to CPU (if not already)
    data, target = data.to('cpu'), target.to('cpu')

    # Make a copy of the data for adversarial processing
    data_copy = data.clone().detach()

    # Generate adversarial example
    adversarial_data = generate_fgsm_adversarial_example(net, data_copy, target, epsilon)

    # Save perturbed image and its label
    perturbed_images.append(adversarial_data.squeeze(0))
    perturbed_labels.append(target)

    # Save images as tensors
    torch.save(adversarial_data, os.path.join(output_dir, f'image{i}_perturbed.pt'))
    torch.save(target, os.path.join(output_dir, f'label{i}.pt'))

    print(f'Image {i+1}/{num_images} processed')

# Save the dataset
perturbed_dataset = {
    'images': perturbed_images,
    'labels': perturbed_labels
}
torch.save(perturbed_dataset, 'perturbed_dataset.pth')

print('Perturbed dataset saved successfully.')
