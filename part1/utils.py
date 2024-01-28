################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"

    epsilon = torch.randn_like(std)

    z = mean + epsilon * std


    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    KLD = 1/2 * torch.sum(torch.exp(2*log_std) + mean**2 - 1 - 2*log_std, dim=-1)
    # raise NotImplementedError

    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """

    elements_in_batch_of_images = torch.prod(torch.tensor(img_shape[1:]))
    bpd = elbo / (elements_in_batch_of_images * torch.log(torch.tensor(2.0)))
    # raise NotImplementedError

    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """
 
    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    ## create percentiles
    percentiles = torch.range(0.5/grid_size, (grid_size-0.5)/grid_size,  step=1/grid_size)

    ## create normal distribution
    normal = torch.distributions.Normal(0, 1)

    ## obtain z values at percentiles
    z1, z2 = torch.meshgrid(percentiles, percentiles, indexing="xy")
    z1 = normal.icdf(percentiles)
    z2 = normal.icdf(percentiles)

    ## concatenate them to use as input for the decoder
    z = torch.stack([z1, z2], dim=-1)
    z = torch.flatten(z, end_dim=-2)

    
    ## apply decoder and softmax to obtain probabilities
    logits = decoder(z)
    probits = torch.nn.functional.softmax(logits, dim=1)

    ## from these probabilities you need to sample using the categorical distribution, same as we do in later defined sample function in train_pl
    ## WHY? because we have as output to decoder a b, 16, h, w image. Thus we need to sample one of those 16 channels based on the probabilities we get
    ## from the decoder. 
    probits = torch.movedim(probits, 1, -1)
    probits = torch.flatten(probits, end_dim=2)
        ## sample from categorical distribution to obtain the most probable value
        
    x_samples = torch.multinomial(probits, 1)

    x_samples = x_samples.reshape(-1, 28, 28, 1)
    x_samples = torch.permute(x_samples, (0, 3, 1, 2))

    ## use the make_grid to combine the grid_size**2 images into a grid and visualize
    img_grid = make_grid(x_samples, nrow=grid_size).float()


    return img_grid

