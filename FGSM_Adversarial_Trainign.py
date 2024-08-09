#Adversarially trained model

import os
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class PerturbedDataset(Dataset):
    def __init__(self, data_dir, num_images):
        self.data_dir = data_dir
        self.num_images = num_images

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, f'image{idx}_perturbed.pt')
        label_path = os.path.join(self.data_dir, f'label{idx}.pt')

        image = torch.load(image_path)
        label = torch.load(label_path)

        # Ensure image has correct shape [channels, height, width]
        if image.dim() == 5:
            image = image.squeeze(1)  # Remove the extra dimension
        elif image.dim() == 2:
            image = image.unsqueeze(0)  # Add the channel dimension if missing

        # Ensure label is a long tensor and squeeze to remove any extra dimension
        label = label.squeeze().long()

        return image, label

# Parameters
data_dir = 'perturbed_images'
num_images = 500  # Number of perturbed images saved

# Create DataLoader
perturbed_dataset = PerturbedDataset(data_dir, num_images)
perturbed_loader = DataLoader(perturbed_dataset, batch_size=64, shuffle=True)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, target in train_loader:
            # Remove any extra dimension from data
            if data.dim() == 5:
                data = data.squeeze(1)

            data, target = data.to('cpu'), target.to('cpu')

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# Load the pretrained ResNet-18 model
net = models.resnet18(pretrained=False, num_classes=10)
net.load_state_dict(torch.load('/content/image_recognition_model.pth', map_location=torch.device('cpu')))
net.train()

# Training the model with adversarial examples
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_model(net, perturbed_loader, criterion, optimizer)

# Save the adversarially trained model
torch.save(net.state_dict(), 'adversarially_trained_model.pth')

print('Model trained and saved successfully.')
