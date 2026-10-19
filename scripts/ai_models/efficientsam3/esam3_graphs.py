"""EfficientSAM3 graph wrappers shared by the SM1 export and comparison scripts.

The point and box path uses the SAM2-style interactive predictor of SAM3 ("tracker" weights):

  PointEncoder : pixel_values [1, 3, 1008, 1008]
                 -> image_embed [1, 256, 72, 72], high_res_feats_0 [1, 32, 288, 288],
                    high_res_feats_1 [1, 64, 144, 144]
  PointDecoder : the three encoder outputs + point_coords [1, N, 2] + point_labels [1, N]
                 + mask_input [1, 1, 288, 288] + has_mask_input [1]
                 -> mask_logits [1, 4, 288, 288], iou_scores [1, 4], object_score_logits [1, 1]

The decoder returns all four mask tokens. The host selects the masks in the same way as
`MaskDecoder.forward`: tokens 1-3 for multimask output, or token 0 with the stability check of
`_dynamic_multimask_via_stability` for single-mask output.

Point coordinates are model-input pixels: normalized image coordinates multiplied by 1008. A box
is two points at the start of the list with labels 2 (top-left) and 3 (bottom-right). Point
labels: 1 positive, 0 negative. The graph appends the padding point (label -1) itself, as
`PromptEncoder.forward` does when no separate box input is given.

The text path uses the detector of the Stage 3 fine-tuned checkpoints:

  TextEncoder      : tokenized_ids [1, 16] -> language_mask, language_features, language_embeds
  DetectorGrounding: pixel_values + the three language tensors
                     -> boxes_xyxy [1, Q, 4] (model-input pixels), scores [1, Q],
                        mask_logits [1, Q, 288, 288]
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

IMAGE_SIZE = 1008
LOW_RES_MASK_SIZE = 288
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
TEXT_CONTEXT_LENGTH = 16


def add_repository_to_path(repository: str) -> None:
    """Make the `sam3` package of the EfficientSAM3 repository importable."""
    package_root = os.path.join(os.path.abspath(repository), "sam3")
    if not os.path.isdir(os.path.join(package_root, "sam3")):
        raise FileNotFoundError(f"no sam3 package under {package_root}")
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


def build_point_model(checkpoint: str, backbone_type: str, model_name: str):
    from sam3.model_builder import build_efficientsam3_image_model

    return build_efficientsam3_image_model(
        checkpoint_path=checkpoint,
        backbone_type=backbone_type,
        model_name=model_name,
        enable_inst_interactivity=True,
        load_from_HF=False,
        device="cpu",
    )


def build_text_model(checkpoint: str, backbone_type: str, model_name: str, text_encoder: str):
    from sam3.model_builder import build_efficientsam3_image_model

    return build_efficientsam3_image_model(
        checkpoint_path=checkpoint,
        backbone_type=backbone_type,
        model_name=model_name,
        text_encoder_type=text_encoder,
        text_encoder_context_length=TEXT_CONTEXT_LENGTH,
        load_from_HF=False,
        device="cpu",
    )


class PointEncoder(nn.Module):
    """Student vision trunk + SAM2 neck + the decoder high-resolution projections."""

    def __init__(self, model) -> None:
        super().__init__()
        vision = model.backbone.vision_backbone
        tracker = model.inst_interactive_predictor.model
        self.trunk = vision.trunk
        self.sam2_convs = vision.sam2_convs
        self.level_count = len(vision.sam2_convs) - model.backbone.scalp
        self.conv_s0 = tracker.sam_mask_decoder.conv_s0
        self.conv_s1 = tracker.sam_mask_decoder.conv_s1
        # no_mem_embed is [1, 1, C] in the (HW, N, C) layout of the tracker.
        self.register_buffer("no_mem_embed", tracker.no_mem_embed.detach().reshape(1, -1, 1, 1))

    def forward(self, pixel_values: torch.Tensor):
        x = self.trunk(pixel_values)[-1]
        levels = [self.sam2_convs[i](x) for i in range(self.level_count)]
        high_res_feats_0 = self.conv_s0(levels[0])
        high_res_feats_1 = self.conv_s1(levels[1])
        image_embed = levels[-1] + self.no_mem_embed
        return image_embed, high_res_feats_0, high_res_feats_1


class PointDecoder(nn.Module):
    """SAM prompt encoder + mask decoder for point and box prompts."""

    def __init__(self, model) -> None:
        super().__init__()
        tracker = model.inst_interactive_predictor.model
        self.prompt_encoder = tracker.sam_prompt_encoder
        self.mask_decoder = tracker.sam_mask_decoder
        self.register_buffer("image_pe", self.prompt_encoder.get_dense_pe().detach())

    def _embed_points(self, coords: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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


class TextEncoder(nn.Module):
    """Student text encoder. The BPE tokenizer runs on the host."""

    def __init__(self, model) -> None:
        super().__init__()
        language = model.backbone.language_backbone
        self.encoder = language.encoder
        self.projector = language.projector

    def forward(self, tokenized_ids: torch.Tensor):
        input_embeds = self.encoder.forward_embedding(tokenized_ids)
        text_memory = self.encoder(input_embeds, return_all_tokens=True, input_is_embeddings=True)
        text_memory = self.projector(text_memory)
        language_mask = tokenized_ids == 0
        return language_mask, text_memory.transpose(0, 1), input_embeds.transpose(0, 1)


class DetectorGrounding(nn.Module):
    """Vision trunk + detector + segmentation head for one text prompt."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values, language_features, language_mask, language_embeds):
        from sam3.model import box_ops
        from sam3.model.data_misc import FindStage

        model = self.model
        backbone_out = model.backbone.forward_image(pixel_values)
        backbone_out["language_features"] = language_features
        backbone_out["language_mask"] = language_mask
        backbone_out["language_embeds"] = language_embeds
        find_stage = FindStage(
            img_ids=torch.zeros(1, dtype=torch.long),
            text_ids=torch.zeros(1, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        outputs = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_stage,
            geometric_prompt=model._get_dummy_prompt(num_prompts=1),
            find_target=None,
        )
        presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        scores = (outputs["pred_logits"].sigmoid() * presence).squeeze(-1)
        boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"]) * IMAGE_SIZE
        return boxes, scores, outputs["pred_masks"]


def preprocess_image(image) -> torch.Tensor:
    """The preprocessing of `Sam3Processor.set_image`: stretch to 1008 x 1008, no letterbox."""
    from torchvision.transforms import v2

    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(NORMALIZE_MEAN), std=list(NORMALIZE_STD)),
    ])
    return transform(v2.functional.to_image(image)).unsqueeze(0)


def tokenize(model, text: str) -> torch.Tensor:
    tokenizer = model.backbone.language_backbone.tokenizer
    return tokenizer([text], context_length=TEXT_CONTEXT_LENGTH)
