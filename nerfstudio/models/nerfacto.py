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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    angular_spectral_loss,  # CHANGE #
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps


@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""    
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_appearance_embedding: bool = True
    """Whether to use an appearance embedding."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    average_init_density: float = 1.0
    """Average initial density output from MLP. """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""
    
    # ============================
    # [CHANGE] Add config option to NerfactoModelConfig class
    # Look for the class definition and add the new field at the end of the list
    # ============================
    num_output_channels: int = 204
    """Number of output channels (RGB=3, HSI=10, etc)."""
    # ============================
    
    # ============================
    # [CHANGE] HSI loss & Angular Spectral loss
    # ============================
    hsi_loss_mult: float = 1.0
    """Weight for spectral reconstruction (HSI) loss."""
    angular_loss_mult: float = 0.0
    """Weight for angular spectral loss. Set >0 to enable."""
    angular_loss_type: Literal["cosine", "acos"] = "cosine"
    """Angular loss form. 'cosine' is stable; 'acos' is SAM-like (radians)."""
    # ============================


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
            
            # [CHANGE 2] Pass config value to Field in populate_modules
            # In 'NerfactoModel' class -> 'populate_modules' method:
            num_output_channels=self.config.num_output_channels,   # <--- [CHANGED] Use config value
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.step = 0
        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        
        # Check
        # print(">> field_outputs[RGB].shape =", field_outputs[FieldHeadNames.RGB].shape)  # 10-band NeRF → [num_rays, num_samples, 10]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict



    # ------------------------------------------
    # [CHANGE] HSI + angular + (mask optional)
    # ------------------------------------------
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)          # [H, W, C]
        pred_image = outputs["rgb"]                     # [H, W, C]
        acc = outputs["accumulation"]                   # [H, W, 1]

        # ------------------------------------------------------------
        # 0) Choose whether to include background in the loss (Nerfstudio-style)
        # ------------------------------------------------------------
        # If you want the original Nerfstudio behavior (background blended), keep this ON.
        # If you want pure full-frame/object-only spectral fidelity without background blending, set to False.
        USE_BLEND_FOR_LOSS = True  # <-- you can later turn this into a config flag if you want

        if USE_BLEND_FOR_LOSS:
            pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
                pred_image=pred_image,
                pred_accumulation=acc,
                gt_image=image,
            )
        else:
            gt_rgb = image
            pred_rgb = pred_image

        # ------------------------------------------------------------
        # 1) Build obj_mask ONCE (mask exists -> object-only, else -> full-frame)
        # ------------------------------------------------------------
        if "mask" in batch:
            obj_mask = batch["mask"].to(self.device)
            if obj_mask.ndim == 3:
                obj_mask = obj_mask[..., 0]             # [H, W]
            obj_mask = obj_mask > 0.5                   # bool [H, W]
        else:
            # no GT mask -> full-frame training
            obj_mask = torch.ones_like(gt_rgb[..., 0], dtype=torch.bool)  # [H, W]

        # empty-mask guard (only matters when GT mask exists but is empty)
        if obj_mask.sum() == 0:
            obj_mask = (acc[..., 0] > 0.01)
        if obj_mask.sum() == 0:
            obj_mask = torch.ones_like(obj_mask, dtype=torch.bool)

        obj_mask_exp = obj_mask.unsqueeze(-1).expand_as(gt_rgb)  # [H, W, C]

        # ------------------------------------------------------------
        # 2) Spectral losses (computed ONCE, with the same mask)
        # ------------------------------------------------------------
        # HSI reconstruction loss (masked if GT mask exists, else full-frame)
        loss_dict["hsi_loss"] = self.config.hsi_loss_mult * self.rgb_loss(
            pred_rgb[obj_mask_exp], gt_rgb[obj_mask_exp]
        )

        # Angular spectral loss (masked if GT mask exists, else full-frame)
        if self.config.angular_loss_mult > 0.0:
            ang = angular_spectral_loss(
                pred=pred_rgb,
                target=gt_rgb,
                mask=obj_mask,
                loss_type=self.config.angular_loss_type,  # "cosine" recommended for training
            )
            loss_dict["ang_loss"] = self.config.angular_loss_mult * ang

        # ------------------------------------------------------------
        # 3) Regular Nerfacto losses (unchanged)
        # ------------------------------------------------------------
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

            if self.config.predict_normals:
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )

            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict



    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        gt_spec = batch["image"].to(self.device)   # [H, W, N]
        pred_spec = outputs["rgb"]                 # [H, W, N]

        acc_map = outputs["accumulation"]          # [H, W, 1]
        fg_mask = acc_map[..., 0] > 0.01           # [H, W]

        # object mask priority: batch["mask"] > fg_mask
        if "mask" in batch:
            obj_mask = batch["mask"].to(self.device)
            if obj_mask.ndim == 3:
                obj_mask = obj_mask[..., 0]
            obj_mask = obj_mask > 0.5
        else:
            obj_mask = fg_mask

        eps = 1e-8

        # ---------- SAM / RMSE (use obj_mask consistently) ----------
        dot = (gt_spec * pred_spec).sum(dim=-1)        # [H, W]
        norm_gt = torch.linalg.norm(gt_spec, dim=-1)   # [H, W]
        norm_pred = torch.linalg.norm(pred_spec, dim=-1)
        cos_theta = dot / (norm_gt * norm_pred + eps)
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        # cosine for training; acos for evaluation
        sam_map = torch.acos(cos_theta)                # [H, W]

        # empty-mask guard for metrics (if mask exists but is empty, fall back to fg_mask; if that's also empty, use full-frame)
        valid_mask = obj_mask
        if valid_mask.sum() == 0:
            valid_mask = fg_mask
        if valid_mask.sum() == 0:
            valid_mask = torch.ones_like(fg_mask, dtype=torch.bool)

        sam_mean_rad = sam_map[valid_mask].mean().item()
        # sam_mean_deg = sam_mean_rad * 180.0 / np.pi

        spec_diff = gt_spec - pred_spec                # [H, W, N]
        spec_mse = (spec_diff ** 2).mean(dim=-1)       # [H, W]
        spec_rmse_map = torch.sqrt(spec_mse + eps)     # [H, W]
        rmse_mean = spec_rmse_map[valid_mask].mean().item()

        # ---------- bbox crop from valid_mask (for image-shaped metrics) ----------
        ys, xs = torch.where(valid_mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        mask_crop = valid_mask[y0:y1, x0:x1]  # [h, w] bool
        if mask_crop.sum() == 0:
            mask_crop = torch.ones_like(mask_crop, dtype=torch.bool)

        # float mask for weighting
        mask_2d_f = mask_crop.to(gt_spec.dtype)  # [h, w]
        mask_sum = mask_2d_f.sum() + eps

        # Helper: get masked mean of SSIM-map
        def _masked_ssim_mean(pred_4d: torch.Tensor, gt_4d: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
            """
            pred_4d, gt_4d: [1, C, h, w] or [1, 1, h, w]
            mask_2d: [h, w] float (0/1)
            returns: scalar tensor
            """
            # Ask torchmetrics to return full SSIM image/map
            out = self.ssim(
                preds=pred_4d,
                target=gt_4d,
                return_full_image=True,
                reduction="none",
            )
            # torchmetrics may return (ssim_val, ssim_img) OR just ssim_img depending on version
            if isinstance(out, (tuple, list)):
                ssim_img = out[1]
            else:
                ssim_img = out

            # Normalize shape to [h, w]
            # Common shapes: [1, h, w] or [1, 1, h, w]
            if ssim_img.ndim == 4:
                ssim_2d = ssim_img[0, 0]  # [h, w]
            elif ssim_img.ndim == 3:
                ssim_2d = ssim_img[0]     # [h, w]
            else:
                # Fallback: if scalar returned for some reason, just return it
                return torch.as_tensor(ssim_img, device=self.device, dtype=pred_4d.dtype)

            m = mask_2d.to(ssim_2d.dtype)
            return (ssim_2d * m).sum() / (m.sum() + eps)

        # ---------- HSI PSNR/SSIM (mask-aware, band-wise mean) ----------
        gt_hsi_crop = gt_spec[y0:y1, x0:x1, :]     # [h, w, N]
        pr_hsi_crop = pred_spec[y0:y1, x0:x1, :]   # [h, w, N]

        gt_hsi = torch.moveaxis(gt_hsi_crop, -1, 0)[None, ...]  # [1, N, h, w]
        pr_hsi = torch.moveaxis(pr_hsi_crop, -1, 0)[None, ...]  # [1, N, h, w]

        psnr_list = []
        ssim_list = []

        # NOTE: PSNR assumes values in [0,1]. If your data isn't normalized, adjust peak value.
        for b in range(gt_hsi.shape[1]):
            g = gt_hsi[:, b:b+1, :, :]  # [1,1,h,w]
            p = pr_hsi[:, b:b+1, :, :]

            # masked PSNR: masked MSE -> PSNR
            diff2 = (p - g) ** 2  # [1,1,h,w]
            mse = (diff2[0, 0] * mask_2d_f).sum() / mask_sum
            psnr_b = 10.0 * torch.log10(1.0 / (mse + eps))
            psnr_list.append(psnr_b)

            # masked SSIM using SSIM-map + masked mean
            ssim_b = _masked_ssim_mean(p, g, mask_2d_f)
            ssim_list.append(ssim_b)

        psnr_hsi = torch.stack(psnr_list).mean()
        ssim_hsi = torch.stack(ssim_list).mean()

        # ---------- pseudo-RGB (visualization + perceptual metrics) ----------
        gt_rgb = gt_spec.clone()
        predicted_rgb = pred_spec.clone()

        if gt_rgb.shape[-1] > 3:
            C = gt_rgb.shape[-1]
            BASE_NUM_BANDS = 204
            band_indices = torch.linspace(
                0, BASE_NUM_BANDS - 1, steps=C, device=gt_rgb.device, dtype=torch.float32
            )
            target_indices = torch.tensor([70.0, 53.0, 19.0], device=gt_rgb.device, dtype=torch.float32)
            diff = torch.abs(target_indices[:, None] - band_indices[None, :])  # [3, C]
            best_channel_idx = torch.argmin(diff, dim=1)                        # [3]
            gt_rgb = gt_rgb[..., best_channel_idx]
            predicted_rgb = predicted_rgb[..., best_channel_idx]

        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)

        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # ---------- RGB metrics (mask-aware) ----------
        gt_crop = gt_rgb[y0:y1, x0:x1, :]                 # [h, w, 3]
        pr_crop = predicted_rgb[y0:y1, x0:x1, :]          # [h, w, 3]

        gt_rgb_for_metrics = torch.moveaxis(gt_crop, -1, 0)[None, ...]  # [1, 3, h, w]
        pr_rgb_for_metrics = torch.moveaxis(pr_crop, -1, 0)[None, ...]  # [1, 3, h, w]

        # masked PSNR for RGB (assumes values in [0,1])
        # diff2_rgb = (pr_rgb_for_metrics - gt_rgb_for_metrics) ** 2  # [1,3,h,w]
        # mse_rgb = (diff2_rgb[0] * mask_2d_f[None, :, :]).sum() / (mask_sum * 3.0)
        # psnr_rgb = 10.0 * torch.log10(1.0 / (mse_rgb + eps))

        # masked SSIM for RGB: SSIM-map averaged over mask
        # ssim_rgb = _masked_ssim_mean(pr_rgb_for_metrics, gt_rgb_for_metrics, mask_2d_f)

        # LPIPS doesn't support masks natively; best practical option:
        # compute LPIPS on bbox crop (keep as-is), or apply the "pred==gt outside mask" trick.
        # Here: pred==gt outside mask trick to reduce background influence.
        mask_4d_rgb = mask_2d_f[None, None, :, :]  # [1,1,h,w]
        pr_rgb_masked = pr_rgb_for_metrics * mask_4d_rgb + gt_rgb_for_metrics * (1.0 - mask_4d_rgb)
        lpips_rgb = self.lpips(pr_rgb_masked, gt_rgb_for_metrics)

        metrics_dict: Dict[str, float] = {
            "hsi_PSNR": float(psnr_hsi.item()),
            "hsi_SSIM": float(ssim_hsi.item()),
            "hsi_SAM_radius": float(sam_mean_rad),
            # "hsi_SAM_degree": float(sam_mean_deg),
            "hsi_RMSE_spectral": float(rmse_mean),

            # "rgb_PSNR": float(psnr_rgb.item()),
            # "rgb_SSIM": float(ssim_rgb.item()) if isinstance(ssim_rgb, torch.Tensor) else float(ssim_rgb),
            "rgb_LPIPS": float(lpips_rgb.item()) if isinstance(lpips_rgb, torch.Tensor) else float(lpips_rgb),
        }

        images_dict: Dict[str, torch.Tensor] = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            images_dict[key] = colormaps.apply_depth_colormap(outputs[key], accumulation=outputs["accumulation"])

        return metrics_dict, images_dict



