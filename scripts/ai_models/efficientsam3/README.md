# EfficientSAM3 export and measurement tools (AI Segmentation Mask, phase SM1)

These scripts export EfficientSAM3 to ONNX and check the export. The Rust program
`rust/puerh_mind/examples/segmentation_runtime_report.rs` then measures the graphs with the
session settings of the `alcedo_mind` sidecar. The plan is
[AI Segmentation Mask master plan, section 8](../../../docs/roadmap/alcedo_studio/edit/ai_segmentation_mask_master_plan.md).

Weights, ONNX files, photos, and reports are not committed. Keep them under `build/tmp/sm1/`.

## Sources

| Item | Value |
| --- | --- |
| Code | `https://github.com/SimonZeng7108/efficientsam3` at commit `bd0936c788fed8d51fa799437f05abd97b401b06` |
| Weights | `https://huggingface.co/Simon7108528/EfficientSAM3` |
| Point and box checkpoint | `stage1_all_converted/efficient_sam3_<backbone>_m_geo_ft.pt` (has the SAM3 tracker point and box decoder) |
| Text checkpoint | `efficientsam3_ft/efficientsam3_<backbone>.pt` (MobileCLIP-S0 student text encoder; no point and box decoder) |
| Tokenizer for the Rust program | `https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json` |

Licenses: the EfficientSAM3 code is Apache-2.0. The checkpoints contain SAM3 weights (the
detector, the SAM3 text encoder in the stage 1 files, and the tracker), so Alcedo Studio treats the
model files as subject to the Meta SAM License. Each published model directory must contain a copy
of both license texts.

Backbone names: EfficientViT-M is `--backbone-type efficientvit --model-name b1`, RepViT-M is
`repvit m1.1`, and TinyViT-M is `tinyvit 11m`.

## Graphs

| File | Inputs | Outputs |
| --- | --- | --- |
| `point_encoder.onnx` | `pixel_values` [1, 3, 1008, 1008] | `image_embed` [1, 256, 72, 72], `high_res_feats_0` [1, 32, 288, 288], `high_res_feats_1` [1, 64, 144, 144] |
| `point_decoder.onnx` | the three encoder outputs, `point_coords` [1, N, 2], `point_labels` [1, N], `mask_input` [1, 1, 288, 288], `has_mask_input` [1] | `mask_logits` [1, 4, 288, 288], `iou_scores` [1, 4], `object_score_logits` [1, 1] |
| `text_encoder.onnx` | `tokenized_ids` [1, 16] int64 | `language_mask` [1, 16] bool, `language_features` [16, 1, 256], `language_embeds` [16, 1, 512] |
| `detector_grounding.onnx` | `pixel_values`, `language_features`, `language_mask` | `boxes_xyxy` [1, 200, 4], `scores` [1, 200], `mask_logits` [1, 200, 288, 288] |

Data rules:

- The image is stretched to 1008 x 1008 without padding (the preprocessing of
  `Sam3Processor.set_image`). Pixel values are `(value / 255 - 0.5) / 0.5`.
- Prompt coordinates are model-input pixels: normalized image coordinates multiplied by 1008.
- A box is two points at the start of the point list with labels 2 and 3. Point labels are 1
  (positive) and 0 (negative). The graph adds the padding point itself.
- The decoder returns the four mask tokens. Multimask output uses tokens 1 to 3. Single-mask
  output uses token 0, or the multimask token with the highest score when the stability score of
  token 0 is below 0.98 (`MaskDecoder._dynamic_multimask_via_stability`).
- Text tokens are CLIP BPE tokens: 49406, the text tokens, 49407, then zero padding to 16.

## Commands

```bash
# Python environment (Git Bash, Windows)
python -m venv build/tmp/sm1/venv
build/tmp/sm1/venv/Scripts/python -m pip install -r scripts/ai_models/efficientsam3/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/SimonZeng7108/efficientsam3 build/tmp/sm1/src/efficientsam3
git -C build/tmp/sm1/src/efficientsam3 checkout bd0936c788fed8d51fa799437f05abd97b401b06

PY="build/tmp/sm1/venv/Scripts/python -I"
REPO=build/tmp/sm1/src/efficientsam3
W=build/tmp/sm1/weights

$PY scripts/ai_models/efficientsam3/inspect_checkpoint.py --repository $REPO \
  --checkpoint $W/efficient_sam3_efficientvit_m_geo_ft.pt \
  --compare-to $W/efficient_sam3_efficientvit_m.pt --out build/tmp/sm1/inspect/geo_ft.json

$PY scripts/ai_models/efficientsam3/export_onnx.py --repository $REPO \
  --point-checkpoint $W/efficient_sam3_efficientvit_m_geo_ft.pt \
  --text-checkpoint $W/efficientsam3_efficientvit.pt \
  --prompts build/tmp/sm1/prompts.json --out-dir build/tmp/sm1/onnx/efficientvit

$PY scripts/ai_models/efficientsam3/compare_onnx_outputs.py --repository $REPO \
  --point-checkpoint $W/efficient_sam3_efficientvit_m_geo_ft.pt \
  --text-checkpoint $W/efficientsam3_efficientvit.pt \
  --onnx-dir build/tmp/sm1/onnx/efficientvit --photos build/tmp/sm1/photos \
  --prompts build/tmp/sm1/prompts.json --out build/tmp/sm1/compare/efficientvit.json

$PY scripts/ai_models/efficientsam3/refine_edges_prototype.py \
  --onnx-dir build/tmp/sm1/onnx/efficientvit --photos build/tmp/sm1/photos \
  --prompts build/tmp/sm1/prompts.json --out-dir build/tmp/sm1/refine

# Quality set (#449): stretch, letterbox, and dual encoder preprocessing
$PY scripts/ai_models/efficientsam3/compare_preprocessing.py \
  --onnx-dir build/tmp/sm1/onnx/efficientvit --photos build/tmp/sm1/quality/photos \
  --prompts build/tmp/sm1/quality/prompts.json \
  --out-dir build/tmp/sm1/quality/preprocess/efficientvit
```

`refine_edges_prototype.py` writes four masks per prompt: (a) bilinear, (b) the gray guided
filter, (c) a second encoder pass on a rectangle around the subject when the rectangle covers less
than half of the photo, and (d) a soft projection matte in a band around the (c) edge. The
`rectangle_iou` value in `refine_timings.json` is the IoU of (a) and (c): the quality cost of the
whole-image encoder input for small subjects (plan section 8.6, cases 5 and 6).

```powershell
# Rust measurement program (PowerShell, Windows). Use --device coreml on macOS.
Set-Location D:\Projects\pu-erh_lab\rust\puerh_mind
cmd /c script\cargo_msvc.cmd build --release --example segmentation_runtime_report
.\target\release\examples\segmentation_runtime_report.exe runtime `
  --onnx-dir ..\..\build\tmp\sm1\onnx\efficientvit --device directml `
  --photos ..\..\build\tmp\sm1\photos --prompts ..\..\build\tmp\sm1\prompts.json `
  --reference-masks ..\..\build\tmp\sm1\compare\reference_masks `
  --tokenizer ..\..\build\tmp\sm1\tokenizer\tokenizer.json `
  --overlay-dir ..\..\build\tmp\sm1\overlays\directml --profile-dir ..\..\build\tmp\sm1\profiles `
  --out ..\..\build\tmp\sm1\runtime\windows_directml.json
# Add --overlay-max-edge 2048 to write smaller overlay images of 40 MP photos.

# gRPC round trip against a running sidecar (alcedo_mind --port 50581)
.\target\release\examples\segmentation_runtime_report.exe ping `
  --endpoint http://127.0.0.1:50581 --out ..\..\build\tmp\sm1\runtime\windows_ping.json
```

## prompts.json

```json
{
  "images": [
    {
      "file": "truck.jpg",
      "points": [{"name": "door", "points": [[0.639, 0.467, 1]]}],
      "boxes": [{"name": "truck", "box": [0.067, 0.233, 0.950, 0.717]}],
      "text": ["car"]
    }
  ]
}
```

Coordinates are normalized image coordinates in `[0, 1]`, x to the right and y down.
