import numpy as np 
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import pickle

from nn_helper import *

# Set seed for reproducibility 
torch.manual_seed(1)
np.random.seed(1)

# Download and load the training set
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])
trainset = torchvision.datasets.MNIST(root="data", download=True, transform = transform)

# Create a DataLoader to efficiently load the data in Instancees
batch_size = 64
data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last = True)

# Use CUDA if possible 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device.")

# Parameters for generator and discriminator 
gen_input_size = 100 
image_size = (28, 28)
n_filters_gen = 128
n_filters_dis = 128

# Make generator and discriminator and move to GPU 
generator = Generator(100, n_filters_gen)
discriminator = Discriminator(n_filters_dis)
generator.to(device)
discriminator.to(device)

# Optimizers for generator and discriminator 
gen_optimizer = torch.optim.Adam(generator.parameters(), lr = 0.0002, betas=(0.5, 0.999))
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 0.0002, betas=(0.5, 0.999))

num_epoch = 100

# Keep track of loss for discriminator and generator and outputs of discriminator
dis_loss_arr = []
gen_loss_arr = []
dis_real_arr = []
dis_fake_arr = []

# Training loop 
for curr_epoch in range(1, num_epoch + 1):
    dis_loss_list = []
    gen_loss_list = []
    dis_real_list = []
    dis_fake_list = [] 
    for batch_real_image, _ in data_loader:
        # Zero the gradients for generator and discriminator 
        generator.zero_grad()
        discriminator.zero_grad()

        # Training generator 
        num_batch_real_image = batch_real_image.shape[0]
        gen_noise = create_noise(num_batch_real_image, gen_input_size).to(device)
        batch_fake_image = generator(gen_noise)
        fake_probs = discriminator(batch_fake_image)
        gen_loss = gan_generator_loss(fake_probs)
        gen_loss.backward()
        gen_optimizer.step()

        # Training discriminator 
        discriminator.zero_grad()

        # Calculate probabilities for real images
        batch_real_image = batch_real_image.to(device)
        real_probs = discriminator(batch_real_image.detach())
            
        # Calculate probabilities for fake images 
        fake_probs = discriminator(batch_fake_image.detach())
        dis_loss = gan_discriminator_loss(real_probs, fake_probs)
        dis_loss.backward()
        dis_optimizer.step()

        # Get useful info  
        # Get loss 
        dis_loss_item = dis_loss.item()
        gen_loss_item = gen_loss.item()
        dis_loss_list.append(dis_loss_item)
        gen_loss_list.append(gen_loss_item) 
        
        # Get probabilities 
        dis_real_list.append(real_probs.mean().item())
        dis_fake_list.append(fake_probs.mean().item())

        del batch_real_image, gen_noise
    # Compute the average loss of the discriminator and generator and probs of discriminator 
    dis_loss = np.mean(dis_loss_list)
    gen_loss = np.mean(gen_loss_list)
    dis_real = np.mean(dis_real_list)
    dis_fake = np.mean(dis_fake_list)

    # Append information to list 
    dis_loss_arr.append(dis_loss)
    gen_loss_arr.append(gen_loss)
    dis_real_arr.append(dis_real)
    dis_fake_arr.append(dis_fake) 
    print(f"Current epoch: {curr_epoch} | Dis_Loss: {dis_loss:3f} | Gen_Loss: {gen_loss:3f} | Fake Probs: {dis_fake:3f} | Real Probs: {dis_real:3f}")
    # Save parameters every 1, 2, 5, 10, and 50 epochs 
    if curr_epoch == 1: 
        torch.save(discriminator.state_dict(), "GAN_params/dis-params-1")
        torch.save(generator.state_dict(), "GAN_params/gen-params-1")
    if curr_epoch == 2: 
        torch.save(discriminator.state_dict(), "GAN_params/dis-params-2")
        torch.save(generator.state_dict(), "GAN_params/gen-params-2")
    if curr_epoch == 5: 
        torch.save(discriminator.state_dict(), "GAN_params/dis-params-5")
        torch.save(generator.state_dict(), "GAN_params/gen-params-5")
    if curr_epoch == 10: 
        torch.save(discriminator.state_dict(), "GAN_params/dis-params-10")
        torch.save(generator.state_dict(), "GAN_params/gen-params-10")
    if curr_epoch == 50: 
        torch.save(discriminator.state_dict(), "GAN_params/dis-params-50")
        torch.save(generator.state_dict(), "GAN_params/gen-params-50")

# Save parameters after training is finished 
torch.save(discriminator.state_dict(), "GAN_params/dis-params-" + str(num_epoch))
torch.save(generator.state_dict(), "GAN_params/gen-params-" + str(num_epoch))

# Save lists after training 
with open('saved_losses/gan_dis_loss_arr.pkl', 'wb') as f:
    pickle.dump(dis_loss_arr, f)

with open('saved_losses/gan_gen_loss_arr.pkl', 'wb') as f:
    pickle.dump(gen_loss_arr, f)

with open('saved_losses/gan_dis_real_arr.pkl', 'wb') as f:
    pickle.dump(dis_real_arr, f)

with open('saved_losses/gan_dis_fake_arr.pkl', 'wb') as f:
    pickle.dump(dis_fake_arr, f)