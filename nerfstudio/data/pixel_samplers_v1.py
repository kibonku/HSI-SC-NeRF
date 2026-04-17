# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for sampling pixels.
"""

import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union

import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.utils.pixel_sampling_utils import divide_rays_per_image, erode_mask


@dataclass
class PixelSamplerConfig(InstantiateConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: PixelSampler)
    """Target class to instantiate."""
    num_rays_per_batch: int = 4096
    """Number of rays to sample per batch."""
    keep_full_image: bool = False
    """Whether or not to include a reference to the full image in returned batch."""
    is_equirectangular: bool = False
    """List of whether or not camera i is equirectangular."""
    ignore_mask: bool = False
    """Whether to ignore the masks when sampling."""
    fisheye_crop_radius: Optional[float] = None
    """Set to the radius (in pixels) for fisheye cameras."""
    rejection_sample_mask: bool = False
    # rejection_sample_mask: bool = True
    # random samples from a complex (Monte Carlo method)
    """Whether or not to use rejection sampling when sampling images with masks"""
    max_num_iterations: int = 100
    """If rejection sampling masks, the maximum number of times to sample"""
    
    # -----------------------
    # [CHANGE] Ray sampling ratio; FG:BG = 80:20
    fg_ratio: float = 0.8
    """Fraction of rays sampled from foreground mask (0~1). Rest is background."""
    # -----------------------


class PixelSampler:
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: PixelSamplerConfig

    def __init__(self, config: PixelSamplerConfig, **kwargs) -> None:
        self.kwargs = kwargs
        self.config = config
        # Possibly override some values if they are present in the kwargs dictionary
        self.config.num_rays_per_batch = self.kwargs.get("num_rays_per_batch", self.config.num_rays_per_batch)
        self.config.keep_full_image = self.kwargs.get("keep_full_image", self.config.keep_full_image)
        self.config.is_equirectangular = self.kwargs.get("is_equirectangular", self.config.is_equirectangular)
        self.config.fisheye_crop_radius = self.kwargs.get("fisheye_crop_radius", self.config.fisheye_crop_radius)
        self.set_num_rays_per_batch(self.config.num_rays_per_batch)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def rejection_sample_mask(
        self,
        mask: Tensor,
        num_samples: int,
        num_images: int,
        image_height: int,
        image_width: int,
        device: Union[torch.device, str],
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Samples pixels within a mask using rejection sampling.

        Args:
            mask: mask of possible pixels in an image to sample from.
            num_samples: number of samples.
            num_images: number of images to sample over.
            image_height: the height of the image.
            image_width: the width of the image.
            device: device that the samples should be on.
        """
        indices = (
            torch.rand((num_samples, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()

        num_valid = 0
        for _ in range(self.config.max_num_iterations):
            c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
            chosen_indices_validity = mask.squeeze(-1)[c, y, x].bool()
            num_valid = int(torch.sum(chosen_indices_validity).item())
            if num_valid == num_samples:
                break
            else:
                replacement_indices = (
                    torch.rand((num_samples - num_valid, 3), device=device)
                    * torch.tensor([num_images, image_height, image_width], device=device)
                ).long()
                indices[~chosen_indices_validity] = replacement_indices

        if num_valid != num_samples:
            """
            ⚠️ Warning은 “ray 샘플링할 때 mask가 너무 작네요” 정도 -> “훈련/샘플링 단계"에 대한 것이지, ns-eval에서 우리가 계산한 SAM/RMSE/PSNR/SSIM 값에는 영향을 미치지 않아.
            Masked sampling failed, mask is either empty or mostly empty.
            Reverting behavior to non-rejection sampling. 
            Consider setting:
            pipeline.datamanager.pixel-sampler.rejection-sample-mask to False > self.config.rejection_sample_mask
            or increasing pipeline.datamanager.pixel-sampler.max-num-iterations > self.config.max_num_iterations
            """
            warnings.warn(
                """ Masked sampling failed, mask is either empty or mostly empty. Reverting behavior to non-rejection sampling. Consider setting pipeline.datamanager.pixel-sampler.rejection-sample-mask to False or increasing pipeline.datamanager.pixel-sampler.max-num-iterations """
            )
            
            # Debuging
            m = mask.squeeze(-1)  # [N,H,W] or [H,W]
            print("nonzero:", int(m.sum().item()))  # nonzero: 7990 > 4096 (self.config.num_rays_per_batch) => Enough!  
            '''전체 512x512=262,144 중 7,990 ≈ 3.0% => 랜덤으로 픽셀 하나 뽑았을 때 “합격(마스크 안)” 확률이 0.03'''

            # self.config.rejection_sample_mask = False
            # nonzero_indices = torch.nonzero(mask.squeeze(), as_tuple=False).to(device)
            # chosen_indices = random.sample(range(len(nonzero_indices)), k=num_samples)
            # indices = nonzero_indices[chosen_indices]
            
            # -----------------------------
            # [CHANGE] Fallback: nonzero 픽셀이 부족하면: replacement 허용해서 뽑아야 함
            # -----------------------------
            nonzero_indices = torch.nonzero(mask.squeeze(-1), as_tuple=False).to(device)  # [M,3] -> (c,y,x)
            n = nonzero_indices.shape[0]

            if n == 0:
                # # truly empty mask -> uniform fallback including background
                indices = (
                    torch.rand((num_samples, 3), device=device)
                    * torch.tensor([num_images, image_height, image_width], device=device)
                ).long()
            else:
                if n >= num_samples:
                    chosen = torch.randperm(n, device=device)[:num_samples]
                    indices = nonzero_indices[chosen]
                else:
                    # allow replacement 
                    chosen = torch.randint(0, n, (num_samples,), device=device)
                    indices = nonzero_indices[chosen]
            # -------------------------------


        return indices

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            if self.config.rejection_sample_mask:
                indices = self.rejection_sample_mask(
                    mask=mask,
                    num_samples=batch_size,
                    num_images=num_images,
                    image_height=image_height,
                    image_width=image_width,
                    device=device,
                )
            else:
                # nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                # chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
                # indices = nonzero_indices[chosen_indices]
                # >>>>>>>>>>> len(nonzero_indices) < batch_size면 ValueError로 바로 터져.
                
                # -----------------------
                # [CHANGE] 
                # -----------------------
                nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False).to(device)  # (c,y,x)
                n = nonzero_indices.shape[0]

                if n == 0:
                    # empty mask -> uniform fallback
                    indices = (
                        torch.rand((batch_size, 3), device=device)
                        * torch.tensor([num_images, image_height, image_width], device=device)
                    ).long()
                elif n >= batch_size:
                    chosen = torch.randperm(n, device=device)[:batch_size]
                    indices = nonzero_indices[chosen]
                else:
                    chosen = torch.randint(0, n, (batch_size,), device=device)  # replacement
                    indices = nonzero_indices[chosen]
                # -----------------------    

        else:
            indices = (
                torch.rand((batch_size, 3), device=device)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def sample_method_equirectangular(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            # Note: if there is a mask, sampling reduces back to uniform sampling, which gives more
            # sampling weight to the poles of the image than the equators.
            # TODO(kevinddchen): implement the correct mask-sampling method.

            indices = self.sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # We sample theta uniformly in [0, 2*pi]
            # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
            # This is done by inverse transform sampling.
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            num_images_rand = torch.rand(batch_size, device=device)
            phi_rand = torch.acos(1 - 2 * torch.rand(batch_size, device=device)) / torch.pi
            theta_rand = torch.rand(batch_size, device=device)
            indices = torch.floor(
                torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def sample_method_fisheye(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            indices = self.sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # Rejection sampling.
            valid: Optional[torch.Tensor] = None
            indices = None
            while True:
                samples_needed = batch_size if valid is None else int(batch_size - torch.sum(valid).item())

                # Check if done!
                if samples_needed == 0:
                    break

                rand_samples = torch.rand((samples_needed, 2), device=device)
                # Convert random samples to radius and theta.
                assert self.config.fisheye_crop_radius is not None
                radii = self.config.fisheye_crop_radius * torch.sqrt(rand_samples[:, 0])
                theta = 2.0 * torch.pi * rand_samples[:, 1]

                # Convert radius and theta to x and y.
                x = (radii * torch.cos(theta) + image_width // 2).long()
                y = (radii * torch.sin(theta) + image_height // 2).long()
                sampled_indices = torch.stack(
                    [torch.randint(0, num_images, size=(samples_needed,), device=device), y, x], dim=-1
                )

                # Update indices.
                if valid is None:
                    indices = sampled_indices
                    valid = (
                        (sampled_indices[:, 1] >= 0)
                        & (sampled_indices[:, 1] < image_height)
                        & (sampled_indices[:, 2] >= 0)
                        & (sampled_indices[:, 2] < image_width)
                    )
                else:
                    assert indices is not None
                    not_valid = ~valid
                    indices[not_valid, :] = sampled_indices
                    valid[not_valid] = (
                        (sampled_indices[:, 1] >= 0)
                        & (sampled_indices[:, 1] < image_height)
                        & (sampled_indices[:, 2] >= 0)
                        & (sampled_indices[:, 2] < image_width)
                    )
            assert indices is not None

        assert indices.shape == (batch_size, 3)
        return indices

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            elif self.config.fisheye_crop_radius is not None:
                indices = self.sample_method_fisheye(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            else:
                indices = self.sample_method(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
        else:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            elif self.config.fisheye_crop_radius is not None:
                indices = self.sample_method_fisheye(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            else:
                indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }
        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = defaultdict(list)

        assert num_rays_per_batch % 2 == 0, "num_rays_per_batch must be divisible by 2"
        num_rays_per_image = divide_rays_per_image(num_rays_per_batch, num_images)

        if "mask" in batch:
            for i, num_rays in enumerate(num_rays_per_image):
                image_height, image_width, _ = batch["image"][i].shape

                indices = self.sample_method(
                    num_rays, 1, image_height, image_width, mask=batch["mask"][i].unsqueeze(0), device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)

                for key, value in batch.items():
                    if key in ["image_idx", "mask"]:
                        continue
                    all_images[key].append(value[i][indices[:, 1], indices[:, 2]])
        else:
            for i, num_rays in enumerate(num_rays_per_image):
                image_height, image_width, _ = batch["image"][i].shape
                if self.config.is_equirectangular:
                    indices = self.sample_method_equirectangular(num_rays, 1, image_height, image_width, device=device)
                else:
                    indices = self.sample_method(num_rays, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                for key, value in batch.items():
                    if key in ["image_idx", "mask"]:
                        continue
                    all_images[key].append(value[i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        collated_batch = {key: torch.cat(all_images[key], dim=0) for key in all_images}

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        c = indices[..., 0].flatten()
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        
        if not hasattr(self, "_printed_once"):
            print("keys:", list(image_batch.keys()))

            if "mask" in image_batch and image_batch["mask"] is not None:
                m = image_batch["mask"]
                if isinstance(m, list):
                    # list of [H,W,1] tensors
                    print("mask is list, len =", len(m))
                    m0 = m[0]
                    print("mask[0]:", m0.shape, m0.dtype, m0.min().item(), m0.max().item())
                else:
                    # tensor [N,H,W,1]
                    print("mask:", m.shape, m.dtype, m.min().item(), m.max().item())
            else:
                print("mask: none")

            self._printed_once = True

        
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch
    
    # ----------------------------
    # [CHANGE] Ray sampling helper
    # ----------------------------
    def _sample_from_bool_mask(
        self,
        mask_bool: Tensor,                 # [N,H,W] bool
        k: int,
        num_images: int,
        image_height: int,
        image_width: int,
        device: Union[torch.device, str], 
        ) -> Int[Tensor, "k 3"]:
        """Return (c,y,x) indices sampled from mask_bool. Safe with replacement + empty handling."""
        nonzero = torch.nonzero(mask_bool, as_tuple=False).to(device)  # [M,3]
        n = nonzero.shape[0]

        if n == 0:
            # fallback: uniform over full image
            return (
                torch.rand((k, 3), device=device)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        if n >= k:
            chosen = torch.randperm(n, device=device)[:k]
            return nonzero[chosen]

        # replacement
        chosen = torch.randint(0, n, (k,), device=device)
        return nonzero[chosen]
    # ----------------------------


@dataclass
class PatchPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PatchPixelSampler."""

    _target: Type = field(default_factory=lambda: PatchPixelSampler)
    """Target class to instantiate."""
    patch_size: int = 32
    """Side length of patch. This must be consistent in the method
    config in order for samples to be reshaped into patches correctly."""
    rejection_sample_mask: bool = True
    """Whether or not to use rejection sampling when sampling images with masks"""
    max_num_iterations: int = 100
    """If rejection sampling masks, the maximum number of times to sample"""


class PatchPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        config: the PatchPixelSamplerConfig used to instantiate class
    """

    config: PatchPixelSamplerConfig

    def __init__(self, config: PatchPixelSamplerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.config.patch_size = self.kwargs.get("patch_size", self.config.patch_size)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.config.patch_size**2)) * (self.config.patch_size**2)

    # overrides base method    
    """ 
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, Tensor) and not self.config.ignore_mask:
            sub_bs = batch_size // (self.config.patch_size**2)
            half_patch_size = int(self.config.patch_size / 2)
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=half_patch_size)

            if self.config.rejection_sample_mask:
                indices = self.rejection_sample_mask(
                    mask=m,
                    num_samples=sub_bs,
                    num_images=num_images,
                    image_height=image_height,
                    image_width=image_width,
                    device=device,
                )
            else:
                nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
                chosen_indices = random.sample(range(len(nonzero_indices)), k=sub_bs)
                indices = nonzero_indices[chosen_indices]

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys - half_patch_size
            indices[:, ..., 2] += xxs - half_patch_size

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)
        else:
            sub_bs = batch_size // (self.config.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.config.patch_size, image_width - self.config.patch_size],
                device=device,
            )

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices
    """
    
    # ---------------------------
    # [CHANGE] Ray sampleing; FG:BG=80:20
    # ---------------------------
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:

        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            # mask: [N,H,W,1] (bool or 0/1)
            fg_k = int(round(batch_size * float(self.config.fg_ratio)))
            bg_k = batch_size - fg_k

            mask_bool = mask[..., 0].bool()          # [N,H,W]
            fg_indices = self._sample_from_bool_mask(
                mask_bool, fg_k, num_images, image_height, image_width, device
            )

            bg_mask_bool = (~mask_bool)              # [N,H,W]
            bg_indices = self._sample_from_bool_mask(
                bg_mask_bool, bg_k, num_images, image_height, image_width, device
            )

            indices = torch.cat([fg_indices, bg_indices], dim=0)

            # optional shuffle so fg/bg are mixed
            perm = torch.randperm(indices.shape[0], device=device)
            indices = indices[perm]
            return indices

        # no mask: uniform
        indices = (
            torch.rand((batch_size, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()
        return indices
    
    # ---------------------------


@dataclass
class PairPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PairPixelSampler."""

    _target: Type = field(default_factory=lambda: PairPixelSampler)
    """Target class to instantiate."""
    radius: int = 2
    """max distance between pairs of pixels."""
    rejection_sample_mask: bool = True
    """Whether or not to use rejection sampling when sampling images with masks"""
    max_num_iterations: int = 100
    """If rejection sampling masks, the maximum number of times to sample"""


class PairPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples pair of pixels from 'image_batch's. Samples pairs of pixels from
        from the images randomly within a 'radius' distance apart. Useful for pair-based losses.

    Args:
        config: the PairPixelSamplerConfig used to instantiate class
    """

    def __init__(self, config: PairPixelSamplerConfig, **kwargs) -> None:
        self.config = config
        self.radius = self.config.radius
        super().__init__(self.config, **kwargs)
        self.rays_to_sample = self.config.num_rays_per_batch // 2

    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: Optional[int],
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        rays_to_sample = self.rays_to_sample
        if batch_size is not None:
            assert int(batch_size) % 2 == 0, (
                f"PairPixelSampler can only return batch sizes in multiples of two (got {batch_size})"
            )
            rays_to_sample = batch_size // 2

        if isinstance(mask, Tensor) and not self.config.ignore_mask:
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=self.radius)

            if self.config.rejection_sample_mask:
                indices = self.rejection_sample_mask(
                    mask=m,
                    num_samples=rays_to_sample,
                    num_images=num_images,
                    image_height=image_height,
                    image_width=image_width,
                    device=device,
                )
            else:
                nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
                chosen_indices = random.sample(range(len(nonzero_indices)), k=rays_to_sample)
                indices = nonzero_indices[chosen_indices]
        else:
            s = (rays_to_sample, 1)
            ns = torch.randint(0, num_images, s, dtype=torch.long, device=device)
            hs = torch.randint(self.radius, image_height - self.radius, s, dtype=torch.long, device=device)
            ws = torch.randint(self.radius, image_width - self.radius, s, dtype=torch.long, device=device)
            indices = torch.concat((ns, hs, ws), dim=1)

        pair_indices = torch.hstack(
            (
                torch.zeros(rays_to_sample, 1, device=device, dtype=torch.long),
                torch.randint(-self.radius, self.radius, (rays_to_sample, 2), device=device, dtype=torch.long),
            )
        )
        pair_indices += indices
        indices = torch.hstack((indices, pair_indices)).view(rays_to_sample * 2, 3)
        return indices
