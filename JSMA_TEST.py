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
net = models.resnet18(pretrained=False, num_classes=10)
net.load_state_dict(torch.load('/content/adversarially_trained_model.pth', map_location=torch.device('cpu')))
net.eval()  # Ensure the model is in evaluation mode

def jsma_attack(model, data, target, epsilon=0.1, num_classes=10):
    data.requires_grad = True
    output = model(data)
    one_hot_target = torch.eye(num_classes)[target].to(data.device)

    # Compute the Jacobian of the output with respect to the input
    gradients = []
    for i in range(num_classes):
        model.zero_grad()
        output[0, i].backward(retain_graph=True)
        gradients.append(data.grad.data.clone())
        data.grad.data.zero_()
    gradients = torch.stack(gradients)  # Shape: [num_classes, 3, 32, 32]

    # Compute the saliency map
    target_grad = gradients[target.item()]  # Shape: [3, 32, 32]
    other_grads = gradients.sum(dim=0)  # Shape: [3, 32, 32]
    saliency_map = target_grad - other_grads  # Shape: [3, 32, 32]

    # Normalize saliency map
    saliency_map = saliency_map / torch.max(torch.abs(saliency_map))

    # Perturb the input image based on the saliency map
    perturbation = epsilon * saliency_map.sign()  # Shape: [3, 32, 32]
    perturbed_data = data + perturbation  # Shape: [1, 3, 32, 32]
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

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
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Parameters
epsilon = 0.01  # Start with small perturbation
num_images = 200  # Number of images to test with perturbation

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

    original_pred = output.argmax(dim=1).item()  # Get the scalar value
    all_original_preds.append(original_pred)
    correct_original += (original_pred == target.item())

    # Generate adversarial example using JSMA
    adversarial_data = jsma_attack(net, data_copy, target, epsilon)

    # Check the shape of adversarial_data
    print(f"Shape of adversarial_data: {adversarial_data.shape}")

    # Ensure adversarial_data has the correct shape
    if adversarial_data.shape != data_copy.shape:
        raise RuntimeError(f"Expected adversarial_data shape {data_copy.shape}, got {adversarial_data.shape}")

    # Adversarial prediction
    output = net(adversarial_data)
    # Check the shape of output
    print(f"Shape of model output (adversarial): {output.shape}")

    # Ensure output is a tensor with one element per batch
    if output.shape != (1, 10):
        raise RuntimeError(f"Expected output shape [1, 10], got {output.shape}")

    adversarial_pred = output.argmax(dim=1).item()  # Get the scalar value from tensor
    all_adversarial_preds.append(adversarial_pred)  # Append the scalar value
    correct_adversarial += (adversarial_pred == target.item())

    # Save original and perturbed images
    original_class = class_names[original_pred]
    adversarial_class = class_names[adversarial_pred]

    image_dir = os.path.join(output_dir, f'image{i}')
    os.makedirs(image_dir, exist_ok=True)

    original_image_path = os.path.join(image_dir, f'image{i}.png')
    adversarial_image_path = os.path.join(image_dir, f'image{i}_perturbed.png')

    save_image_with_classification(data, f'Original: {original_class}', original_image_path)
    save_image_with_classification(adversarial_data, f'Adversarial: {adversarial_class}', adversarial_image_path)

    print(f'Image {i+1}/{num_images}')
    print('Original prediction:', original_class)
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
