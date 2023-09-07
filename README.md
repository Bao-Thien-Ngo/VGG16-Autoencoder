# VGG16-Autoencoder

Applying gamma correction to the RESISC-45 dataset to make the images darker. Here is the link to the dataset: https://mega.nz/fm/yttEQJTJ.

After that, I modified the VGG16 model by adding the latent space of the autoencoder to the middle of the convolution layers of the VGG16 model. The image below is the VGG16 + Autoencoder model:

   ![image](https://github.com/Bao-Thien-Ngo/VGG16-Autoencoder/assets/79235839/bc2edeb0-11ce-416b-9233-23b5e3fd9bb4)
   

Then, I split the dataset into train, valid, and test datasets with the ratio 70:20:10. After I trained the model for 100 epochs, I got an accuracy of 69.62%  for the testing dataset.
