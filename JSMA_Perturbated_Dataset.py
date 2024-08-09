import os
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the pretrained ResNet-18 model
net = models.resnet18(pretrained=False, num_classes=10)
net.load_state_dict(torch.load('/content/image_recognition_model.pth', map_location=torch.device('cpu')))
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

def ensure_correct_shape(tensor):
    if tensor.dim() == 5:
        tensor = tensor.squeeze(1)  # Remove the extra dimension
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)  # Add the channel dimension if missing
    return tensor

# Data transformation and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# Parameters
epsilon = 0.01  # Start with small perturbation
num_images = 500  # Number of images to test with perturbation

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

    # Generate adversarial example using JSMA
    adversarial_data = jsma_attack(net, data_copy, target, epsilon)

    # Ensure the correct shape of the perturbed data
    adversarial_data = ensure_correct_shape(adversarial_data)

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
