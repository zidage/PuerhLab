"""BiRefNet graph wrapper for the AI Segmentation Mask model re-evaluation (#458, step E2).

One graph per variant:

  SubjectMatte : pixel_values [1, 3, S, S] -> alpha [1, 1, S, S]

S is 1024 (lite, general, matting) or 2048 (the high-resolution variants). `alpha` is the sigmoid
of the last BiRefNet output, the foreground matte in [0, 1] at the model input size. The host
resizes it to the photo size with bilinear interpolation.

The preprocessing is the one of the model cards: stretch the whole photo to S x S (no padding),
then `(value / 255 - mean) / std` with the ImageNet mean and std.

The model code is `birefnet.py` and `BiRefNet_config.py` of the Hugging Face repository of each
variant (the repository code with the backbone set for that variant). The scripts load it with
`transformers` and `trust_remote_code` from a local directory that also holds `model.safetensors`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# Variant name -> Hugging Face repository, input size, backbone, training task.
VARIANTS = {
    "lite": ("ZhengPeng7/BiRefNet_lite", 1024, "swin_v1_t", "general"),
    "general": ("ZhengPeng7/BiRefNet", 1024, "swin_v1_l", "general"),
    "hr": ("ZhengPeng7/BiRefNet_HR", 2048, "swin_v1_l", "general"),
    "lite_matting": ("ZhengPeng7/BiRefNet_lite-matting", 1024, "swin_v1_t", "matting"),
    "matting": ("ZhengPeng7/BiRefNet-matting", 1024, "swin_v1_l", "matting"),
    "hr_matting": ("ZhengPeng7/BiRefNet_HR-matting", 2048, "swin_v1_l", "matting"),
}


def variant_directory(weights_root: str, variant: str) -> Path:
    """`<weights_root>/<repository name>`: the Hugging Face files of one variant."""
    repository, _, _, _ = VARIANTS[variant]
    return Path(weights_root) / repository.split("/", 1)[1]


def image_size(variant: str) -> int:
    return VARIANTS[variant][1]


def build_model(weights_root: str, variant: str) -> nn.Module:
    """The BiRefNet model of one variant in float32 on the CPU, in eval mode."""
    from transformers import AutoModelForImageSegmentation

    directory = variant_directory(weights_root, variant)
    if not (directory / "model.safetensors").is_file():
        raise FileNotFoundError(f"no model.safetensors under {directory}")
    model = AutoModelForImageSegmentation.from_pretrained(str(directory), trust_remote_code=True)
    return model.float().eval()


def deform_conv2d_grid_sample(input, offset, weight, bias=None, stride=(1, 1), padding=(0, 0),
                              dilation=(1, 1), mask=None):
    """`torchvision.ops.deform_conv2d` (one offset group, one weight group) as one bilinear
    `grid_sample` and one 1 x 1 convolution per kernel point, accumulated.

    The ONNX graph then has only standard operators (GridSample, Conv) and the largest
    intermediate tensor has the size of the input. The decomposition of `deform_conv2d_onnx_exporter`
    gathers all kernel points at once (822 MB per tensor for the 7 x 7 kernel at 256 x 256 in
    BiRefNet lite), which DirectML cannot allocate.

    Sampling position of output (y, x) and kernel point (i, j), as in torchvision:
    `y * stride - padding + i * dilation + offset[2k]` (rows) and the same with `offset[2k + 1]`
    (columns), bilinear, zero outside the input. `grid_sample` with `align_corners=True` and zero
    padding gives the same values.
    """
    pair = nn.modules.utils._pair
    (sh, sw), (ph, pw), (dh, dw) = pair(stride), pair(padding), pair(dilation)
    _, _, height, width = input.shape
    out_channels, in_channels, kh, kw = weight.shape
    if in_channels != input.shape[1] or offset.shape[1] != 2 * kh * kw:
        raise ValueError("only one offset group and one weight group are supported")
    out_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    base_y = (torch.arange(out_h, dtype=input.dtype) * sh - ph).view(1, out_h, 1)
    base_x = (torch.arange(out_w, dtype=input.dtype) * sw - pw).view(1, 1, out_w)
    output = None
    for i in range(kh):
        for j in range(kw):
            k = i * kw + j
            y = base_y + i * dh + offset[:, 2 * k]
            x = base_x + j * dw + offset[:, 2 * k + 1]
            grid = torch.stack((x * (2.0 / (width - 1)) - 1.0, y * (2.0 / (height - 1)) - 1.0),
                               dim=-1)
            sampled = nn.functional.grid_sample(input, grid, mode="bilinear",
                                                padding_mode="zeros", align_corners=True)
            if mask is not None:
                sampled = sampled * mask[:, k:k + 1]
            term = nn.functional.conv2d(sampled, weight[:, :, i:i + 1, j:j + 1])
            output = term if output is None else output + term
    if bias is not None:
        output = output + bias.view(1, out_channels, 1, 1)
    return output


def use_grid_sample_deform_conv(model: nn.Module) -> None:
    """Replace `deform_conv2d` in the module of the BiRefNet code with the grid_sample form."""
    import sys

    module = sys.modules[type(model).__module__]
    if not hasattr(module, "deform_conv2d"):
        raise AttributeError(f"{module.__name__} has no deform_conv2d")
    module.deform_conv2d = deform_conv2d_grid_sample


def check_grid_sample_deform_conv(seed: int = 0) -> float:
    """Largest absolute difference between torchvision `deform_conv2d` and the grid_sample form,
    relative to the largest absolute torchvision output, on random inputs with the kernel sizes
    and paddings of BiRefNet (1, 3, 7). Float32 rounding of the normalized grid coordinates gives
    about 3e-6."""
    from torchvision.ops import deform_conv2d

    generator = torch.Generator().manual_seed(seed)
    worst = 0.0
    for kernel in (1, 3, 7):
        x = torch.randn(1, 8, 23, 31, generator=generator)
        offset = 3.0 * torch.randn(1, 2 * kernel * kernel, 23, 31, generator=generator)
        mask = torch.rand(1, kernel * kernel, 23, 31, generator=generator) * 2.0
        weight = torch.randn(5, 8, kernel, kernel, generator=generator)
        bias = torch.randn(5, generator=generator)
        padding = kernel // 2
        reference = deform_conv2d(x, offset, weight, bias, padding=padding, mask=mask)
        candidate = deform_conv2d_grid_sample(x, offset, weight, bias, padding=(padding, padding),
                                              mask=mask)
        worst = max(worst, float((reference - candidate).abs().max() / reference.abs().max()))
    return worst


class SubjectMatte(nn.Module):
    """`pixel_values` -> `alpha`: the sigmoid of the last decoder output."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)[-1].sigmoid()


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess(image: Image.Image, size: int) -> torch.Tensor:
    """`transforms.Resize((size, size))` on the PIL image (bilinear with antialias), then the
    ImageNet normalization, in NCHW order."""
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.array(NORMALIZE_MEAN, np.float32)) / np.array(NORMALIZE_STD, np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


def alpha_to_photo(alpha: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize of an S x S alpha to the photo size (`align_corners=False`)."""
    tensor = torch.from_numpy(np.ascontiguousarray(alpha, np.float32))[None, None]
    resized = torch.nn.functional.interpolate(tensor, size=(height, width), mode="bilinear",
                                              align_corners=False)
    return resized[0, 0].clamp(0.0, 1.0).numpy()
