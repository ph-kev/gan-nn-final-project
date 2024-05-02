import numpy as np 
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

from nn_helper import *

torch.manual_seed(1)

# Download and load the training set
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])
trainset = torchvision.datasets.MNIST(root="data", download=True, transform = transform)

# Create a DataLoader to efficiently load the data in Instancees
batch_size = 64
data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last = True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device.")

gen_input_size = 100 
image_size = (28, 28)
n_filters = 32 

generator = Generator(100, n_filters)
discriminator = Discriminator(n_filters)

generator.to(device)
discriminator.to(device)

gen_optimizer = torch.optim.RMSprop(generator.parameters(), lr = 5e-5)
dis_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr = 5e-5)

num_epoch = 500
for curr_epoch in range(num_epoch):
    for batch_real_image, _ in data_loader:
        num_batch_real_image = batch_real_image.shape[0]
        for i in range(0, 5):
            # Training discriminator 
            discriminator.zero_grad()

            # Calculate probabilities for real images
            batch_real_image = batch_real_image.to(device)
            dis_real_probs = discriminator(batch_real_image)
            
            # Calculate probabilities for fake images 
            dis_noise = create_noise(num_batch_real_image, gen_input_size).to(device)
            batch_fake_image = generator(dis_noise)
            dis_fake_probs = discriminator(batch_fake_image)
            dis_loss = wgan_discriminator_loss(dis_real_probs, dis_fake_probs)
            dis_loss.backward()
            dis_optimizer.step()
            for param in discriminator.parameters():
                param.data.clamp_(-0.01, 0.01)

        # Training generator 
        generator.zero_grad()
        discriminator.zero_grad()
        gen_noise = create_noise(num_batch_real_image, gen_input_size).to(device)
        batch_fake_image = generator(gen_noise)
        fake_probs = discriminator(batch_fake_image)
        gen_loss = wgan_generator_loss(fake_probs)
        gen_loss.backward()
        gen_optimizer.step()

        # Print useful information 
        dis_loss_item = dis_loss.item()
        gen_loss_item = gen_loss.item()
        print(f"Current epoch: {curr_epoch} | Dis_Loss: {dis_loss_item:3f} | Gen_Loss: {gen_loss_item:3f}")
        del batch_real_image, gen_noise, dis_noise
    # Save parameters every 5 epochs 
    # Save parameters every 1, 2, 5, 10, and 50 epochs 
    if curr_epoch == 1: 
        torch.save(discriminator.state_dict(), "WGAN_params/dis-params-1")
        torch.save(generator.state_dict(), "WGAN_params/gen-params-1")
    if curr_epoch == 2: 
        torch.save(discriminator.state_dict(), "WGAN_params/dis-params-2")
        torch.save(generator.state_dict(), "WGAN_params/gen-params-2")
    if curr_epoch == 5: 
        torch.save(discriminator.state_dict(), "WGAN_params/dis-params-5")
        torch.save(generator.state_dict(), "WGAN_params/gen-params-5")
    if curr_epoch == 10: 
        torch.save(discriminator.state_dict(), "WGAN_params/dis-params-10")
        torch.save(generator.state_dict(), "WGAN_params/gen-params-10")
    if curr_epoch == 50: 
        torch.save(discriminator.state_dict(), "WGAN_params/dis-params-50")
        torch.save(generator.state_dict(), "WGAN_params/gen-params-50")

# Save parameters after training is finished 
torch.save(discriminator.state_dict(), "WGAN_params/dis-params-" + str(num_epoch))
torch.save(generator.state_dict(), "WGAN_params/gen-params-" + str(num_epoch))