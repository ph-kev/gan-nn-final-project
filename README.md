# GAN Final Project for CS152 

## Repository explanation 
The code implement GAN and WGAN and tested them on the MNIST dataset. We also computed the FID and SSIM scores on the generated images. 

## File structure 
`GAN_params` - GAN parameters (`dis-params-n` is the paramters for the discriminator after n epochs and `gen-params-n` is the paramters for the generator after n epochs).

`plots` - Saved plots of loss and discrimiantor output. 

`saved_loss` - Saved losses as NumPy arrays.

`WGAN_params` - GAN parameters (`dis-params-n` is the paramters for the discriminator after n epochs and `gen-params-n` is the paramters for the generator after n epochs).

`evaluating_GAN_generator.ipynb` - Evaluate GAN using FID and SSIM scores. 

`evaluating_WGAN_generator.ipynb` - Evaluate WGAN using FID and SSIM scores. 

`GAN.py` - Python script to train GAN. 

`nn_helper.py` - Helper file that store the classes for the generator and discriminator and loss functions. 

`plotting.ipynb` - Python notebook for plotting the losses, discriminator outputs, and generated images. 

`WGAN.py` - Python script to train WGAN. 
