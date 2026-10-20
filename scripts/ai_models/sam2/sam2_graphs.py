"""SAM 2.1 graph wrappers for the AI Segmentation Mask model re-evaluation (#458, step E1).

The graphs have the same inputs and outputs as the EfficientSAM3 point and box graphs of SM1
(`scripts/ai_models/efficientsam3/esam3_graphs.py`), so `segmentation_runtime_report` runs them
without a code path per model:

  PointEncoder : pixel_values [1, 3, 1024, 1024]
                 -> image_embed [1, 256, 64, 64], high_res_feats_0 [1, 32, 256, 256],
                    high_res_feats_1 [1, 64, 128, 128]
  PointDecoder : the three encoder outputs + point_coords [1, N, 2] + point_labels [1, N]
                 + mask_input [1, 1, 256, 256] + has_mask_input [1]
                 -> mask_logits [1, 4, 256, 256], iou_scores [1, 4], object_score_logits [1, 1]
  CandidateDecoder (#458, step E4): the three encoder outputs + point_coords [B, 1, 2]
                 -> mask_logits [B, 3, 256, 256], iou_scores [B, 3], stability_scores [B, 3]

The encoder is `SAM2ImagePredictor.set_image`: the Hiera trunk, the FPN neck, the `conv_s0` and
`conv_s1` projections of the mask decoder, and `no_mem_embed` on the stride-16 map. The decoder
returns all four mask tokens; the host selects the masks as `MaskDecoder.forward` does (tokens
1-3 for multimask output, token 0 with the stability check for single-mask output).

Point coordinates are model-input pixels: normalized image coordinates multiplied by 1024. A box
is two points at the start of the list with labels 2 and 3. The graph appends the padding point.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

IMAGE_SIZE = 1024
LOW_RES_MASK_SIZE = 256
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# Variant name -> (config name inside the sam2 package, checkpoint file name).
VARIANTS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


def add_repository_to_path(repository: str) -> None:
    """Make the `sam2` package of the SAM 2 repository importable without installing it."""
    root = os.path.abspath(repository)
    if not os.path.isfile(os.path.join(root, "sam2", "build_sam.py")):
        raise FileNotFoundError(f"no sam2 package under {root}")
    if root not in sys.path:
        sys.path.insert(0, root)


def build_model(variant: str, checkpoint: str):
    """`build_sam2` with the default post-processing overrides (dynamic multimask selection)."""
    from sam2.build_sam import build_sam2

    config, _ = VARIANTS[variant]
    return build_sam2(config, checkpoint, device="cpu", mode="eval")


class PointEncoder(nn.Module):
    """Hiera trunk + FPN neck + decoder high-resolution projections + `no_mem_embed`."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        # no_mem_embed is [1, 1, C] in the (HW, N, C) layout of the video model.
        self.register_buffer("no_mem_embed", model.no_mem_embed.detach().reshape(1, -1, 1, 1))

    def forward(self, pixel_values: torch.Tensor):
        backbone_out = self.model.forward_image(pixel_values)
        levels = backbone_out["backbone_fpn"][-self.model.num_feature_levels:]
        high_res_feats_0, high_res_feats_1, image_embed = levels
        if self.model.directly_add_no_mem_embed:
            image_embed = image_embed + self.no_mem_embed
        return image_embed, high_res_feats_0, high_res_feats_1


class PointDecoder(nn.Module):
    """SAM 2.1 prompt encoder + mask decoder for point and box prompts."""

    def __init__(self, model) -> None:
        super().__init__()
        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder
        self.register_buffer("image_pe", self.prompt_encoder.get_dense_pe().detach())

    def _embed_points(self, coords: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # `PromptEncoder._embed_points(pad=True)` without the data-dependent label loop.
        encoder = self.prompt_encoder
        coords = coords + 0.5
        padding_point = torch.zeros_like(coords[:, :1, :])
        padding_label = -torch.ones_like(labels[:, :1])
        coords = torch.cat([coords, padding_point], dim=1)
        labels = torch.cat([labels, padding_label], dim=1)
        size = encoder.input_image_size
        normalized = torch.stack([coords[..., 0] / size[1], coords[..., 1] / size[0]], dim=-1)
        embedding = encoder.pe_layer._pe_encoding(normalized)
        labels = labels.unsqueeze(-1)
        embedding = torch.where(labels == -1, torch.zeros_like(embedding)
                                + encoder.not_a_point_embed.weight, embedding)
        for label in range(4):
            embedding = torch.where(labels == label,
                                    embedding + encoder.point_embeddings[label].weight, embedding)
        return embedding

    def forward(self, image_embed, high_res_feats_0, high_res_feats_1, point_coords,
                point_labels, mask_input, has_mask_input):
        sparse = self._embed_points(point_coords, point_labels)
        encoder = self.prompt_encoder
        mask_embedding = encoder.mask_downscaling(mask_input)
        no_mask_embedding = encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        has_mask = has_mask_input.reshape(1, 1, 1, 1)
        dense = has_mask * mask_embedding + (1.0 - has_mask) * no_mask_embedding
        mask_logits, iou_scores, _, object_score_logits = self.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=[high_res_feats_0, high_res_feats_1],
        )
        return mask_logits, iou_scores, object_score_logits


class CandidateDecoder(nn.Module):
    """The point grid of `SAM2AutomaticMaskGenerator`: one positive point per prompt, B prompts.

    The graph is `SAM2ImagePredictor._predict` for B single-point prompts with
    `multimask_output=True` and `repeat_image=True` (the batched path of the generator):

      image_embed [1, 256, 64, 64], high_res_feats_0 [1, 32, 256, 256],
      high_res_feats_1 [1, 64, 128, 128], point_coords [B, 1, 2]
      -> mask_logits [B, 3, 256, 256] (tokens 1-3, clamped to +-32 as the predictor does),
         iou_scores [B, 3], stability_scores [B, 3]

    `stability_scores` is `calculate_stability_score` (threshold 0, offset 1.0) on the 256 x 256
    logits. The generator computes it on the masks at the photo size; the host can then filter
    the candidates without reading every logit plane first.
    """

    def __init__(self, model, stability_offset: float = 1.0) -> None:
        super().__init__()
        self.point_decoder = PointDecoder(model)
        self.stability_offset = stability_offset

    def forward(self, image_embed, high_res_feats_0, high_res_feats_1, point_coords):
        decoder = self.point_decoder
        labels = torch.ones_like(point_coords[..., 0])
        sparse = decoder._embed_points(point_coords, labels)
        # [1, C, 1, 1]: the no-mask embedding broadcasts over the B repeated image embeddings.
        dense = decoder.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        mask_logits, iou_scores, _, _ = decoder.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=decoder.image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=True,
            high_res_features=[high_res_feats_0, high_res_feats_1],
        )
        mask_logits = torch.clamp(mask_logits[:, 1:], -32.0, 32.0)
        iou_scores = iou_scores[:, 1:]
        inner = (mask_logits > self.stability_offset).flatten(-2).sum(-1).float()
        outer = (mask_logits > -self.stability_offset).flatten(-2).sum(-1).float()
        stability_scores = torch.where(outer > 0, inner / outer.clamp(min=1.0),
                                       torch.ones_like(outer))
        return mask_logits, iou_scores, stability_scores


def grid_points(points_per_side: int) -> torch.Tensor:
    """`build_point_grid`: points_per_side**2 normalized points, row by row, cell centres."""
    offset = 1.0 / (2 * points_per_side)
    side = torch.linspace(offset, 1.0 - offset, points_per_side)
    ys, xs = torch.meshgrid(side, side, indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


def preprocess_image(image) -> torch.Tensor:
    """The preprocessing of `SAM2Transforms`: stretch to 1024 x 1024, ImageNet normalization."""
    from torchvision.transforms import Normalize, Resize, ToTensor

    tensor = ToTensor()(image)
    tensor = Resize((IMAGE_SIZE, IMAGE_SIZE))(tensor)
    tensor = Normalize(list(NORMALIZE_MEAN), list(NORMALIZE_STD))(tensor)
    return tensor.unsqueeze(0)


def prompt_sets(entry: dict) -> list[dict]:
    """Point and box prompt sets of one prompts.json entry in normalized coordinates."""
    sets = []
    for item in entry.get("points", []):
        sets.append({"name": item["name"], "points": [[p[0], p[1]] for p in item["points"]],
                     "labels": [p[2] for p in item["points"]], "box": None})
    for item in entry.get("boxes", []):
        sets.append({"name": item["name"], "points": [], "labels": [], "box": item["box"]})
    return sets


def model_prompt(prompt: dict, size: int = IMAGE_SIZE):
    """Model-input coordinates and labels: a box first (labels 2 and 3), then the points."""
    coords, labels = [], []
    if prompt["box"] is not None:
        x0, y0, x1, y1 = prompt["box"]
        coords += [[x0 * size, y0 * size], [x1 * size, y1 * size]]
        labels += [2, 3]
    coords += [[x * size, y * size] for x, y in prompt["points"]]
    labels += prompt["labels"]
    return coords, labels


def uses_multimask(prompt: dict) -> bool:
    """One point without a box is ambiguous: use the multimask output, as SM1 does."""
    return prompt["box"] is None and len(prompt["points"]) == 1
