import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

base_path = "/Users/kyong/Documents/DSA3101/proj"
red_path = os.path.join(base_path, 'red')
blue_path = os.path.join(base_path, 'blue')
green_path = os.path.join(base_path, 'green')

for root, dirs, files in os.walk('/Users/kyong/Documents/DSA3101/proj'):
    for file in files:
        if file == ".DS_Store":
            os.remove(os.path.join(root, file))

#display the first image of each folder in 128x128
def display_first_image(folder_path, size=(128, 128)):
  for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
      img_path = os.path.join(folder_path, filename)
      img = Image.open(img_path)
      img = img.resize(size)
      plt.imshow(img)
      plt.title(f"First image in {folder_path}")
      plt.show()
      break  # Display only the first image

# Display the first image from each folder
'''display_first_image(red_path)
display_first_image(blue_path)
display_first_image(green_path)'''

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert to tensor
])

# Custom dataset class for conditional image loading
class ColorTopDataset(Dataset):
    def __init__(self, red_path, blue_path, green_path, transform=None):
        self.red_images = [os.path.join(red_path, img) for img in os.listdir(red_path)]
        self.blue_images = [os.path.join(blue_path, img) for img in os.listdir(blue_path)]
        self.green_images = [os.path.join(green_path, img) for img in os.listdir(green_path)]
        self.all_images = self.red_images + self.blue_images + self.green_images
        self.labels = [0] * len(self.red_images) + [1] * len(self.blue_images) + [2] * len(self.green_images)
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Create dataset and dataloader
dataset = ColorTopDataset(red_path, blue_path, green_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.ReLU(True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        flattened_img_size = int(np.prod(img_shape))

        self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img = img[:, :3, :, :]
        d_in = torch.cat((img.view(img.size(0), -1), self.label_emb(labels)), -1)
        validity = self.model(d_in)
        return validity
    
# Parameters
latent_dim = 100
img_shape = (3, 128, 128)
num_classes = 3  # Red, Blue, Green
learning_rate_G = 0.0002
learning_rate_D = 0.000007
n_epochs = 100

# Initialize generator and discriminator
generator = Generator(latent_dim, num_classes, img_shape)
discriminator = Discriminator(num_classes, img_shape)

# Optimizers and loss function
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate_G)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_D)
adversarial_loss = nn.BCELoss()

# Training loop
for epoch in range(n_epochs):
    for imgs, labels in dataloader:

        batch_size = imgs.size(0)

        # Adversarial ground truths
        valid = torch.ones((batch_size, 1), requires_grad=False)
        fake = torch.zeros((batch_size, 1), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, latent_dim)
        gen_labels = torch.randint(0, num_classes, (batch_size,))

        # Generate images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real images
        real_loss = adversarial_loss(discriminator(imgs, labels), valid)

        # Loss for fake images
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch+1}/{n_epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

# Generate a sample image for each color label (0=Red, 1=Blue, 2=Green)
generator.eval()

for color_label in range(num_classes):
    z = torch.randn(1, latent_dim)
    label = torch.tensor([color_label])

    with torch.no_grad():
        gen_img = generator(z, label)

    # Rescale image from [-1, 1] to [0, 1]
    gen_img = 0.5 * gen_img + 0.5
    plt.imshow(gen_img.squeeze(0).permute(1, 2, 0).cpu())
    plt.title(f"Generated Top Color: {['Red', 'Blue', 'Green'][color_label]}")
    plt.axis("off")
    plt.show()    
