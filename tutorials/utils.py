import torch
import matplotlib
from typing import Literal
import cv2
import numpy as np

def apply_float_colormap(image, colormap: Literal["turbo", "grey"] = "turbo", non_zero: bool = False):
    # colormap = "turbo"
    # image = image[..., None]
    if non_zero:
        image = image - torch.min(image[image != 0])
    else:
        image = image - torch.min(image)
    image = image / (torch.max(image) + 1e-5)
    image = torch.clip(image, 0, 1)
    image = torch.nan_to_num(image, 0)
    # print(image.shape)
    if colormap == "grey":
        # return image.repeat(1, 1, 3)
        image = image.expand(*image.shape[:-1], 3).contiguous()
        # print(image.shape)
        # exit()
        return image
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"

    return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[
        image_long[..., 0]
    ]


def pix2world(uv, depth, intr, extr):
    """
    convert uv and depth to world coordinates
    uv: (N, 2)
    depth: (N, 1)
    intr: (4,)
    extr: (3, 4)

    return: (N, 3)
    """
    # Create the intrinsic matrix K
    K = torch.eye(3).cuda()
    K[0, 0] = intr[0]
    K[1, 1] = intr[1]
    K[0, 2] = intr[2]
    K[1, 2] = intr[3]
    
    # Invert the intrinsic matrix
    K_inv = torch.inverse(K)
    
    # Prepare the uv coordinates (adjust from image to normalized camera coordinates)
    uv_hom = torch.cat([uv + 0.5, torch.ones(uv.shape[0], 1).cuda()], dim=1)
    uv_norm = uv_hom * depth
    
    # Convert image coordinates to camera coordinates
    pt_cam = torch.matmul(K_inv, uv_norm.t())
    
    # Invert the extrinsic matrix
    R = extr[:3, :3]
    t = extr[:3, -1].unsqueeze(dim=1)
    
    # Inverse rotation and translation
    R_inv = torch.inverse(R)
    
    # Compute world coordinates
    xyz = torch.matmul(R_inv, pt_cam[:3] - t)
    return xyz.t()

def complex_texture_sampling(gt_image, gt_depth, num_points=5000):
    # read the image
    image = gt_image.cpu().numpy()*255
    H, W = image.shape[:2]

    # convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # compute the gradient of the image
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # add uniform magnitude to the gradient magnitude, to avoid zero probability
    gradient_magnitude = gradient_magnitude + np.min(gradient_magnitude[gradient_magnitude>0])

    # mediate the gradient magnitude
    probability_distribution = gradient_magnitude / np.sum(gradient_magnitude)

    # sample points from the probability distribution
    sampled_points = np.random.choice(np.arange(gray_image.size), size=num_points, p=probability_distribution.flatten())

    # convert the sampled points to coordinates on the image
    sampled_coordinates = np.unravel_index(sampled_points, gray_image.shape)

    XY = image.shape[:2][::-1]
    xys = np.array(sampled_coordinates).T[:,::-1].copy() # (num_points, 2) 2 is x, y, corresponding to W, H

    depths_norm = gt_depth[sampled_coordinates]
    scales = 1 / probability_distribution[sampled_coordinates]
    scales_norm = 0.5 * scales / np.max(scales)
    rgbs = image[sampled_coordinates]
    rgbs_norm = rgbs / 255.

    return xys, depths_norm, scales_norm, rgbs_norm