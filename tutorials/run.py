
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import imageio
from datetime import datetime
import os
import render
import utils

class SimpleGaussian:
    def __init__(self, gt_image, gt_depth, num_points=100000, depth_scale=10.0, depth_offset=5.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gt_image = gt_image.to(self.device)
        self.gt_depth = gt_depth.to(self.device) if gt_depth is not None else None
        self.num_points = num_points
        self.depth_scale = depth_scale
        self.depth_offset = depth_offset

        H, W, C = gt_image.shape 
        self.H, self.W = H, W
    
        self.bg = 0. # black
        fov = math.pi / 2.0
        fx = 0.5 * float(W) / math.tan(0.5 * fov)
        fy = 0.5 * float(H) / math.tan(0.5 * fov)
        self.intr = torch.Tensor([fx, fy, float(W) / 2, float(H) / 2]).cuda().float()
        self.extr = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 1.0]]).cuda().float()
        
        N = int(num_points)
        self._attributes = {
            "xyz":      torch.rand((N, 3), dtype=torch.float32).cuda() * 2 - 1,
            "scale":    torch.rand((N, 3), dtype=torch.float32).cuda(),
            "rotate":   torch.rand((N, 4), dtype=torch.float32).cuda(),
            "opacity":  torch.rand((N, 1), dtype=torch.float32).cuda(),
            "rgb":      torch.rand((N, 3), dtype=torch.float32).cuda()
        }

        # let the z value to be zero
        # self._attributes["xyz"][:,2] = 0.
        
        self._activations = {
            "scale": lambda x: torch.abs(x) + 1e-8,
            "rotate": torch.nn.functional.normalize,
            "opacity": torch.sigmoid,
            "rgb": torch.sigmoid
        }

        # the inverse of the activation functions
        self._activations_inv = {
            "scale": lambda x: torch.abs(x),
            "rotate": torch.nn.functional.normalize,
            "opacity": torch.logit,
            "rgb": torch.logit
        }
        
        # logs
        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        # Create a directory with the current date and time as its name
        directory = f"logs/{now}"
        os.makedirs(directory, exist_ok=True)
        self.dir = directory

    def prior_init(self, gt_image, gt_depth, num_points=None):
        if num_points is None:
            num_points = self.num_points
        xys, depths, scales, rgbs = utils.complex_texture_sampling(gt_image, gt_depth, num_points=num_points)
        
        xys = torch.from_numpy(xys).to(self.device).float()
        depths = depths.to(self.device).float() * self.depth_scale + self.depth_offset
        self.gt_depth = gt_depth.to(self.device).float() * self.depth_scale + self.depth_offset
    
        self._attributes["xyz"] = utils.pix2world(xys, depths, self.intr, self.extr)
        print("[init] x range: ", self._attributes["xyz"][:,0].min().item(), self._attributes["xyz"][:,0].max().item())
        print("[init] y range: ", self._attributes["xyz"][:,1].min().item(), self._attributes["xyz"][:,1].max().item())
        print("[init] z range: ", self._attributes["xyz"][:,2].min().item(), self._attributes["xyz"][:,2].max().item())

        # fn = lambda x: np.power(x, 0.6)
        # fn = lambda x: np.sqrt(x)
        # fn = lambda x: x
        # self.fn = fn
        # scales = self.fn(scales)
        scales = scales * (depths/depths.min()).squeeze().cpu().numpy()
        scales = torch.from_numpy(scales).float().unsqueeze(1).repeat(1, 3).to(self.device)
        # scales = torch.clamp(scales, max=1e-2)
        self._attributes["scale"] = self._activations_inv["scale"](scales)

        rgbs = torch.from_numpy(rgbs).float().contiguous().to(self.device)
        eps = 1e-15  # avoid logit function input 0 or 1
        rgbs = torch.clamp(rgbs, min=eps, max=1-eps)
        # calculate the inverse of sigmoid function, i.e., logit function
        self._attributes["rgb"] = self._activations_inv["rgb"](rgbs)

        # opacity = torch.ones((num_points, 1), device=self.device).float() - eps
        # self._attributes["opacity"] = self._activations_inv["opacity"](opacity)

    def train(self, max_iter=1000, lr=1e-2, lambda_depth=0., save_imgs=False, save_videos=False):
        frames = []
        frames_depth = []
        frames_center = []
        progress_bar = tqdm(range(1, max_iter), desc="Training")
        l1_loss = nn.SmoothL1Loss(reduce="none")
        mse_loss = nn.MSELoss()
        mse_loss_pixel = nn.MSELoss(reduction='none')

        # optim
        for attribute_name in self._attributes.keys():
            self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)
        
        self.optimizer = torch.optim.Adam(list(self._attributes.values()), lr=lr)

        # training
        for iteration in range(0, max_iter):
            loss = 0.

            input_group = [
                self.get_attribute("xyz"),
                self.get_attribute("scale"),
                self.get_attribute("rotate"),
                self.get_attribute("opacity"),
                self.get_attribute("rgb"),
                self.intr,
                self.extr,
                self.bg,
                self.W,
                self.H,
            ]

            return_dict = render.render_multiple(
                input_group,
                ["rgb", "uv", "depth", "depth_map", "depth_map_color", "center"]
            )

            # render image
            rendered_rgb, uv, depth = return_dict["rgb"], return_dict["uv"], return_dict["depth"]
            
            # render depth map
            rendered_depth_map = return_dict["depth_map"]

            # render colorful depth map
            rendered_depth_map_color = return_dict["depth_map_color"]
            
            # render center
            rendered_center = return_dict["center"]
            
            loss_rgb_pixel = mse_loss_pixel(rendered_rgb.permute(1, 2, 0), self.gt_image).mean(dim=2)
            loss_rgb = loss_rgb_pixel.mean()
            loss += loss_rgb
            progress_dict = {"rgb": f"{loss_rgb.item():.6f}"}

            rendered_depth_map = rendered_depth_map.permute(1, 2, 0)

            if lambda_depth > 0:
                loss_depth = mse_loss_pixel(rendered_depth_map, self.gt_depth) / (self.gt_depth + rendered_depth_map)
                loss_depth = loss_depth.mean()

                loss += lambda_depth * loss_depth
                progress_dict["depth"] = f"{loss_depth.item():.6f}"

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            progress_dict["total"] = loss.item()
            progress_bar.set_postfix(progress_dict)
            progress_bar.update(1)

            if iteration % 10 == 0:
                # rgb
                rendered_rgb_np = render.render2img(rendered_rgb)
                frames.append(rendered_rgb_np)
                # depth map
                rendered_depth_map_color_np = render.render2img(rendered_depth_map_color)
                frames_depth.append(rendered_depth_map_color_np)
                # center
                rendered_center_np = render.render2img(rendered_center)
                frames_center.append(rendered_center_np)
      
        progress_bar.close()
        print(rendered_depth_map[0,0])
        print(rendered_depth_map[-1,-1])

        if save_imgs:
            os.makedirs(os.path.join(self.dir, "images"), exist_ok=True)
            imageio.imwrite(os.path.join(self.dir, "images", f"img.png"), rendered_rgb_np)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_center.png"), rendered_center_np)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_depth.png"), rendered_depth_map_color_np)

        if save_videos:
            # save them as a video with imageio
            frames_np = np.stack(frames, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_rgb.mp4"), frames_np, fps=30)
            frames_center_np = np.stack(frames_center, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_center.mp4"), frames_center_np, fps=30)
            frames_depth_np = np.stack(frames_depth, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_depth.mp4"), frames_depth_np, fps=30)

    def get_attribute(self, name):
        try:
            if name in self._activations.keys() and self._activations[name] is not None:
                return self._activations[name](self._attributes[name])
            else:
                return self._attributes[name]
        except:
            raise ValueError(f"Attribute or activation for {name} is not VALID!")

if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)

    # read image
    img = imageio.imread("./data/face.png") # H, W, C
    img = img.astype(np.float32) / 255.0
    gt_image = torch.from_numpy(img).cuda()
    print("image shape: ", gt_image.shape)
    
    # read depth
    depth = imageio.imread("./data/face_depth.png") # H, W, C
    depth = np.expand_dims(depth, axis=-1)
    depth = depth.astype(np.float32) / 255.0
    gt_depth = torch.from_numpy(depth).cuda()
    gt_depth = gt_depth.mean(dim=-1, keepdim=True)
    # depths: 0(black) is far, 1(white) is near
    gt_depth = 1. - gt_depth
    # depths: 0 is near, 1 is far
    print("depth shape: ", gt_depth.shape)

    # w/ prior-driven initialization
    gaussians = SimpleGaussian(num_points=10000, gt_image=gt_image, gt_depth=gt_depth)
    gaussians.prior_init(gt_image=gt_image, gt_depth=gt_depth)
    gaussians.train(max_iter=500, lr=1e-2, lambda_depth=0.1, save_imgs=True, save_videos=True)

    # w/o prior-driven initialization
    gaussians = SimpleGaussian(num_points=10000, gt_image=gt_image, gt_depth=gt_depth)
    gaussians.train(max_iter=5000, lr=1e-2, save_imgs=True, save_videos=True)
    