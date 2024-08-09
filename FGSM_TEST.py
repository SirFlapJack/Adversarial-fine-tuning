#FGSM attack

#original images tested on the model then apply perturbation and test them again

#we have cat_01.png, we test this original image on the model then we apply perturbation and thest the model agan as cat_perturbed_01.png



import os

import torchvision.models as models

import torch

import torch.nn as nn

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import confusion_matrix

import seaborn as sns



# Load the pretrained ResNet-18 model

net = models.resnet18(weights=False, num_classes=10)

net.load_state_dict(torch.load('/content/adversarially_trained_model.pth', map_location=torch.device('cpu')))

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



def save_image_with_classification(image_data, classification, filepath):

    image = np.transpose(image_data.squeeze(0).detach().numpy(), (1, 2, 0))

    image = np.clip(image, 0, 1)

    plt.imshow(image)

    plt.title(classification)

    plt.axis('off')

    plt.savefig(filepath)

    plt.close()



# Load class names

class_names = datasets.CIFAR10(root='./data', train=True, download=True).classes



# Data transformation and loading

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

test_loader = DataLoader(test_data, batch_size=1, shuffle=True)



# Parameters

epsilon = 0.01  # Start with zero perturbation to test

num_images = 100  # Number of images to test with perturbation



# Create directory to save images

output_dir = 'perturbed_images'

os.makedirs(output_dir, exist_ok=True)



# Initialize lists to collect true labels and predictions

all_labels = []

all_original_preds = []

all_adversarial_preds = []



correct_original = 0

correct_adversarial = 0



for i, (data, target) in enumerate(test_loader):

    if i >= num_images:

        break



    # Move data and target to CPU (if not already)

    data, target = data.to('cpu'), target.to('cpu')



    # Append the true label to the list

    all_labels.append(target.item())



    # Make a copy of the data for adversarial processing

    data_copy = data.clone().detach()



    output = net(data_copy)

    original_pred = output.argmax(dim=1, keepdim=True)

    all_original_preds.append(original_pred.item())

    correct_original += original_pred.eq(target.view_as(original_pred)).sum().item()



    # Generate adversarial example

    adversarial_data = generate_fgsm_adversarial_example(net, data_copy, target, epsilon)



    # Adversarial prediction

    output = net(adversarial_data)

    adversarial_pred = output.argmax(dim=1, keepdim=True)

    all_adversarial_preds.append(adversarial_pred.item())

    correct_adversarial += adversarial_pred.eq(target.view_as(adversarial_pred)).sum().item()



    # Save original and perturbed images

    original_class = class_names[original_pred.item()]

    adversarial_class = class_names[adversarial_pred.item()]



    image_dir = os.path.join(output_dir, f'image{i}')

    os.makedirs(image_dir, exist_ok=True)



    original_image_path = os.path.join(image_dir, f'image{i}.png')

    adversarial_image_path = os.path.join(image_dir, f'image{i}_perturbed.png')



    save_image_with_classification(data, f'Original: {original_class}', original_image_path)

    save_image_with_classification(adversarial_data, f'Adversarial: {adversarial_class}', adversarial_image_path)



    print(f'Image {i+1}/{num_images}')

    print('Original prediction:',  original_class)

    print('Adversarial prediction:', adversarial_class)



# Calculate accuracy

accuracy_original = correct_original / num_images

accuracy_adversarial = correct_adversarial / num_images



print(f'\nAccuracy on original images: {accuracy_original * 100:.2f}%')

print(f'Accuracy on adversarial images: {accuracy_adversarial * 100:.2f}%')



# Generate confusion matrices

conf_matrix_original = confusion_matrix(all_labels, all_original_preds, labels=range(10))

conf_matrix_adversarial = confusion_matrix(all_labels, all_adversarial_preds, labels=range(10))



# Plot confusion matrices

plt.figure(figsize=(10, 8))

sns.heatmap(conf_matrix_original, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

plt.title('Confusion Matrix for Original Images')

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()



plt.figure(figsize=(10, 8))

sns.heatmap(conf_matrix_adversarial, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)

plt.title('Confusion Matrix for Adversarial Images')

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
