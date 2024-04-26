import numpy as np 
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

torch.manual_seed(1)

# Download and load the training set
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])
trainset = torchvision.datasets.MNIST(root="data", download=True, transform = transform)

# Create a DataLoader to efficiently load the data in Instancees
batch_size = 64
data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Define class for generator and discriminator 
# https://arxiv.org/pdf/1511.06434.pdf 
# The paper about deep convolutional generative adversarial networks (DCGAN) give 
# advice about making DCGANs. It is basically the same template as here. 
class Generator(nn.Module):
    def __init__(self, input_size, n_filters):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(input_size, n_filters*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_filters*4, n_filters*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        'Forward pass'
        output = self.network(x)
        return output

class Discriminator(nn.Module):
    def __init__(self, n_filters):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters, n_filters*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters*2, n_filters*4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        'Forward pass'
        output = self.network(x)
        return output.view(-1, 1).squeeze(0) 

# Generate noise from normal distribution 
def create_noise(batch_size, z_size):
    input_z = torch.normal(0.0, 1.0, (batch_size, z_size, 1, 1))
    return input_z

# The loss function is from WGAN (https://arxiv.org/abs/1701.07875)
def discriminator_loss(real_output, fake_output):
  # num_real_samples = real_output.shape[0]
  # num_fake_samples = fake_output.shape[0]
  # loss = -torch.sum((1 / num_real_samples) * (torch.log(real_output)) + (1 / num_fake_samples) * (torch.log(1.0 - fake_output)))
  loss = -(real_output.mean() - fake_output.mean())
  return loss

def generator_loss(fake_output):
  # num_fake_samples = fake_output.shape[0]
  # loss = -torch.sum((1 / num_fake_samples) * (torch.log(fake_output)))
  loss = -fake_output.mean()
  return loss 

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
        for i in range(0, 5):
            # Training discriminator 
            discriminator.zero_grad()

            # Calculate probabilities for real images
            batch_real_image = batch_real_image.to(device)
            dis_real_probs = discriminator(batch_real_image)
            
            # Calculate probabilities for fake images 
            num_batch_real_image = batch_real_image.shape[0]
            dis_noise = create_noise(num_batch_real_image, gen_input_size).to(device)
            batch_fake_image = generator(dis_noise)
            dis_fake_probs = discriminator(batch_fake_image)
            dis_loss = discriminator_loss(dis_real_probs, dis_fake_probs)
            dis_loss.backward()
            dis_optimizer.step()
            for param in discriminator.parameters():
                param.data.clamp_(-0.01, 0.01)

        # Training generator 
        generator.zero_grad()
        gen_noise = create_noise(num_batch_real_image, gen_input_size).to(device)
        batch_fake_image = generator(gen_noise)
        fake_probs = discriminator(batch_fake_image)
        gen_loss = generator_loss(fake_probs)
        gen_loss.backward()
        gen_optimizer.step()

        # Print useful information 
        dis_loss_item = dis_loss.item()
        gen_loss_item = gen_loss.item()
        print(f"Current epoch: {curr_epoch} | Dis_Loss: {dis_loss_item:3f} | Gen_Loss: {gen_loss_item:3f}")
        del batch_real_image, gen_noise, dis_noise
    # Save parameters every 5 epochs so that code will run smoothly 
    if curr_epoch % 5 == 0:
        torch.save(discriminator.state_dict(), "dis-params-" + str(num_epoch))
        torch.save(generator.state_dict(), "gen-params-" + str(num_epoch))

# Save parameters after training is finished 
torch.save(discriminator.state_dict(), "WGAN_params/dis-params-" + str(num_epoch))
torch.save(generator.state_dict(), "WGAN_params/gen-params-" + str(num_epoch))