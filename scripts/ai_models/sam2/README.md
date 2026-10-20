# SAM 2.1 export and measurement tools (AI Segmentation Mask, #458 steps E1, E3, and E4)

These scripts export the SAM 2.1 point and box path to ONNX, check the export against PyTorch, and
make the evidence of the model re-evaluation. The Rust program
`rust/puerh_mind/examples/segmentation_runtime_report.rs` measures the graphs with the session
settings of the `alcedo_mind` sidecar. The reports are
[E1: SAM 2.1 point and box prompts](../../../docs/roadmap/alcedo_studio/edit/ai_mask_candidate/e1_sam2_1_point_box_evaluation.md),
[E3: stroke selection](../../../docs/roadmap/alcedo_studio/edit/ai_mask_candidate/e3_stroke_selection.md),
and
[E4: click selection with precomputed candidates](../../../docs/roadmap/alcedo_studio/edit/ai_mask_candidate/e4_click_candidates.md);
the plan is [AI Segmentation Mask master plan, section 8.16](../../../docs/roadmap/alcedo_studio/edit/ai_segmentation_mask_master_plan.md).

Weights, ONNX files, photos, and reports are not committed. Keep them under `build/tmp/explore/`.

## Sources

| Item | Value |
| --- | --- |
| Code | `https://github.com/facebookresearch/sam2` at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| Weights | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_<tiny\|small\|base_plus\|large>.pt` |
| License | Apache-2.0 for the code and the checkpoints (`LICENSE` in the repository, 11,357 bytes, SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`) |

## Files

| File | Purpose |
| --- | --- |
| `sam2_graphs.py` | `PointEncoder` and `PointDecoder` wrappers with the tensor layout of the SM1 EfficientSAM3 graphs; preprocessing and prompt helpers. |
| `export_onnx.py` | Export `point_encoder.onnx`, `point_decoder.onnx`, and `export_report.json` for each variant. |
| `compare_onnx_outputs.py` | `OnnxMaskLogitsMatchPytorchWithinTolerance`: `SAM2ImagePredictor` compared with ONNX Runtime CPU; writes the reference masks for the Rust program. |
| `probe_masks.py` | Uncertainty, grid energy, hole, and island metrics of plan section 8.16.1 for any point and box graph pair (SAM 2.1 or EfficientSAM3). |
| `run_with_gpu_memory.ps1` | Run a program and record its peak dedicated and shared GPU memory from the Windows `GPU Process Memory` counters. |
| `make_comparison_sheets.py` | Overlay, logit, and detail-crop comparison sheets for the report. |
| `compare_candidate_outputs.py` | E4: `CandidateDecoderLogitsMatchPytorchWithinTolerance`, the candidate graph compared with `SAM2ImagePredictor._predict` on the generator point grid; optionally the masks of the upstream `SAM2AutomaticMaskGenerator`. |
| `click_targets.json` | E4: the intended selection (whole object or named part) and the E1 reference mask of each single-point prompt of the quality set. |
| `probe_candidates.py` | E4: tables of the candidate reports (time to candidates, coverage, selection per prompt, hover result) and the comparison with the upstream generator. |
| `make_candidate_sheets.py` | E4: candidate maps and the levels under the pointer per prompt. |
| `compare_stroke_outputs.py` | E3: `StrokePromptLogitsMatchPytorchWithinTolerance`, every corrective sequence of a `strokes` report replayed with `SAM2ImagePredictor.predict` and ONNX Runtime CPU (several points, negative points, `mask_input`); with `--pad-points`, the padded prompts on both sides. |
| `probe_strokes.py` | E3: tables of the stroke reports (first interaction, length and point count, spacing, direction, corrective sequences, decoder time, agreement with a second report). |
| `make_stroke_sheets.py` | E3: one click, strokes, and the corrective sequences per prompt, with the strokes and points drawn. |

## Graphs

| File | Inputs | Outputs |
| --- | --- | --- |
| `point_encoder.onnx` | `pixel_values` [1, 3, 1024, 1024] | `image_embed` [1, 256, 64, 64], `high_res_feats_0` [1, 32, 256, 256], `high_res_feats_1` [1, 64, 128, 128] |
| `point_decoder.onnx` | the three encoder outputs, `point_coords` [1, N, 2], `point_labels` [1, N], `mask_input` [1, 1, 256, 256], `has_mask_input` [1] | `mask_logits` [1, 4, 256, 256], `iou_scores` [1, 4], `object_score_logits` [1, 1] |
| `candidate_decoder.onnx` (E4) | the three encoder outputs, `point_coords` [B, 1, 2] (one positive point per prompt) | `mask_logits` [B, 3, 256, 256] (tokens 1-3, clamped to +-32), `iou_scores` [B, 3], `stability_scores` [B, 3] |

Data rules:

- The image is stretched to 1024 x 1024 without padding (`SAM2Transforms`). Pixel values are
  `(value / 255 - mean) / std` with the ImageNet mean `(0.485, 0.456, 0.406)` and std
  `(0.229, 0.224, 0.225)`. `export_report.json` states these values in `model_input`; the Rust
  program reads them from there.
- Prompt coordinates are model-input pixels: normalized image coordinates multiplied by 1024. A
  box is two points at the start of the list with labels 2 and 3. The graph adds the padding point.
- The decoder returns the four mask tokens. The mask selection is the same as for EfficientSAM3:
  tokens 1 to 3 for multimask output; token 0, or the best multimask token when the stability
  score of token 0 is below 0.98, for single-mask output.
- `point_decoder.onnx` is different for each variant (each checkpoint trains its own decoder).
- `candidate_decoder.onnx` is the batched point grid of `SAM2AutomaticMaskGenerator`
  (`multimask_output=True`, `repeat_image=True`). `stability_scores` is
  `calculate_stability_score` (threshold 0, offset 1.0) on the 256 x 256 logits; the upstream
  generator computes it on the masks at the photo size. `--candidate-batch 64` fixes B to 64 (the
  whole graph is then one DirectML node); `--candidate-float16` converts the graph to float16
  inside, with float32 inputs and outputs. The E4 measurements use both options.

## Commands

```bash
# Python environment and sources (Git Bash, Windows)
E=build/tmp/explore
python -m venv $E/venv
$E/venv/Scripts/python -m pip install -r scripts/ai_models/sam2/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/facebookresearch/sam2 $E/src/sam2
git -C $E/src/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
for m in tiny small base_plus large; do
  curl -L -o $E/weights/sam2.1_hiera_$m.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_$m.pt
done

PY="$E/venv/Scripts/python -I"
Q=build/tmp/sm1/quality
$PY scripts/ai_models/sam2/export_onnx.py --repository $E/src/sam2 --weights $E/weights \
  --variants tiny,small,base_plus,large --out-root $E/e1/onnx
for v in tiny small base_plus large; do
  $PY scripts/ai_models/sam2/compare_onnx_outputs.py --repository $E/src/sam2 --weights $E/weights \
    --variant $v --onnx-dir $E/e1/onnx/$v --photos $Q/photos --prompts $Q/prompts.json \
    --out $E/e1/compare/$v/compare.json
  $PY scripts/ai_models/sam2/probe_masks.py --onnx-dir $E/e1/onnx/$v --label "SAM 2.1 $v" \
    --photos $Q/photos --prompts $Q/prompts.json --logit-dir $E/e1/probe/logits/$v \
    --out $E/e1/probe/$v.json
done
# The same probe on the SM1 EfficientSAM3 graphs. --grid-lines 72,144,32,64,96 reproduces the
# grid energy values of plan section 8.16.1.
$PY scripts/ai_models/sam2/probe_masks.py --onnx-dir build/tmp/sm1/onnx/efficientvit \
  --label "EfficientSAM3 EfficientViT-M" --photos $Q/photos --prompts $Q/prompts.json \
  --logit-dir $E/e1/probe/logits/esam3_efficientvit --out $E/e1/probe/esam3_efficientvit.json
```

```powershell
# DirectML measurement with GPU memory (PowerShell, Windows). Repeat for each variant.
Set-Location D:\Projects\pu-erh_lab
$e = "build\tmp\explore\e1"; $q = "build\tmp\sm1\quality"
.\scripts\ai_models\sam2\run_with_gpu_memory.ps1 `
  -Exe rust\puerh_mind\target\release\examples\segmentation_runtime_report.exe `
  -Arguments @('runtime', '--onnx-dir', "$e\onnx\tiny", '--device', 'directml',
               '--photos', "$q\photos", '--prompts', "$q\prompts.json",
               '--reference-masks', "$e\compare\tiny\reference_masks",
               '--overlay-dir', "$e\overlays\tiny", '--overlay-max-edge', '2048',
               '--profile-dir', "$e\profiles\tiny", '--profile-graphs', 'point_encoder,point_decoder',
               '--out', "$e\runtime\tiny_directml.json") `
  -Out "$e\runtime\tiny_directml_gpu_memory.json" -Log "$e\runtime\tiny_directml.log"
```

The profiling pass loads a second session while the first one is still loaded. Record the GPU
memory from a run without `--profile-dir`.

```bash
# Comparison sheets of the report. The output directory is in .gitignore: the images show photos
# that must not be shared again, so they are not distributed with the GitHub repository.
$PY scripts/ai_models/sam2/make_comparison_sheets.py \
  --model "EfficientSAM3 EffViT-M=$E/e1/overlays/esam3_efficientvit|$E/e1/probe/logits/esam3_efficientvit" \
  --model "SAM 2.1 tiny=$E/e1/overlays/tiny|$E/e1/probe/logits/tiny" \
  --model "SAM 2.1 small=$E/e1/overlays/small|$E/e1/probe/logits/small" \
  --model "SAM 2.1 base+=$E/e1/overlays/base_plus|$E/e1/probe/logits/base_plus" \
  --model "SAM 2.1 large=$E/e1/overlays/large|$E/e1/probe/logits/large" \
  --prompts $Q/prompts.json --quality 72 \
  --out-dir docs/roadmap/alcedo_studio/edit/ai_mask_candidate/images/e1_sam2_1 \
  --logit-prompt DSC00674.jpg:person --logit-prompt DSC00674.jpg:face \
  --logit-prompt DSC00728.jpg:sky --logit-prompt Nikon-D800e-raw-00007.jpg:sky \
  --logit-prompt DSC_1188.jpg:tail --logit-prompt DSC00602.jpg:tram \
  --crop "DSC00674.jpg:person:0.15,0.0,0.95,0.45" --crop "DSC00326.jpg:person:0.36,0.6,0.68,0.9" \
  --crop "DSC00936.jpg:person:0.47,0.63,0.68,0.93" --crop "DSC_1188.jpg:squirrel:0.3,0.1,0.72,0.64" \
  --crop "2088064441.jpg:boarder:0.4,0.45,0.62,0.66" --crop "DSC00562.jpg:ducks_3pt:0.33,0.35,0.56,0.52" \
  --crop "DSC00602.jpg:tram_box:0.35,0.6,0.92,0.92"
```

### E4: click candidates and the live point decoder

Run one command at a time. The PyTorch checks use 4 threads (about 2 to 5 GB of memory); the
DirectML runs use the GPU. The input set has the 4096 px long edge of the application input.

```bash
E=build/tmp/explore; PY="$E/venv/Scripts/python -I"; Q=build/tmp/sm1/quality
# Photos and E1 reference masks at the application input size.
$PY scripts/ai_models/sam2/make_input_set.py --photos $Q/photos \
  --reference-masks $E/e1/compare/base_plus/reference_masks --max-edge 4096 --out $E/e4/input_4096
# Graphs: the point graphs of E1 and a candidate graph with a fixed batch of 16 prompts (float32).
# --candidate-float16 makes the float16 option of the report.
for v in tiny base_plus; do
  $PY scripts/ai_models/sam2/export_onnx.py --repository $E/src/sam2 --weights $E/weights \
    --variants $v --candidate-batch 16 --out-root $E/e4/onnx_float32_batch16
done
# CandidateDecoderLogitsMatchPytorchWithinTolerance. --upstream-out also runs the upstream
# generator on 1024 px copies (tiny only in the report).
for v in tiny base_plus; do
  $PY scripts/ai_models/sam2/compare_candidate_outputs.py --repository $E/src/sam2 \
    --weights $E/weights --variant $v --onnx-dir $E/e4/onnx_float32_batch16/$v \
    --photos $E/e4/input_4096/photos --batches-per-photo 2 --threads 4 \
    --out $E/e4/compare/${v}_float32.json
done
```

```powershell
# DirectML run per variant (nested setting of the report). The upstream setting is
# --score-threshold 0.8 --stability-threshold 0.95 --dedupe box --nms-threshold 0.7.
Set-Location D:\Projects\pu-erh_lab
$e = "build\tmp\explore"; $i = "$e\e4\input_4096"
.\scripts\ai_models\sam2\run_with_gpu_memory.ps1 `
  -Exe rust\puerh_mind\target\release\examples\segmentation_runtime_report.exe `
  -Arguments @('candidates', '--onnx-dir', "$e\e4\onnx_float32_batch16\tiny", '--device', 'directml',
               '--photos', "$i\photos", '--prompts', 'build\tmp\sm1\quality\prompts.json',
               '--targets', 'scripts\ai_models\sam2\click_targets.json',
               '--reference-masks', "$i\reference_masks",
               '--reference-scores', "$e\e1\compare\base_plus\compare.json",
               '--points-per-side', '16,32', '--batch', '16', '--batch-sizes', '16', '--runs', '5',
               '--score-threshold', '0.5', '--stability-threshold', '0.85', '--dedupe', 'mask',
               '--nms-threshold', '0.85', '--overlay-dir', "$e\e4\overlays\tiny",
               '--overlay-points-per-side', '32', '--out', "$e\e4\runtime\tiny_directml.json") `
  -Out "$e\e4\runtime\tiny_directml_gpu_memory.json" -Log "$e\e4\runtime\tiny_directml.log"
```

```bash
# Tables and sheets of the report. The sheets are in .gitignore (photos that must not be shared).
$PY scripts/ai_models/sam2/probe_candidates.py --report "tiny=$E/e4/runtime/tiny_directml.json" \
  --report "base plus=$E/e4/runtime/base_plus_directml.json" --out $E/e4/probe/final.json
$PY scripts/ai_models/sam2/make_candidate_sheets.py \
  --model "SAM 2.1 tiny=$E/e4/runtime/tiny_directml.json|$E/e4/overlays/tiny" \
  --model "SAM 2.1 base plus=$E/e4/runtime/base_plus_directml.json|$E/e4/overlays/base_plus" \
  --grid 32 --out-dir docs/roadmap/alcedo_studio/edit/ai_mask_candidate/images/e4_click
```

### E3: stroke selection

The E3 runs use the E4 input set and the E4 graphs (`point_encoder.onnx` and `point_decoder.onnx`
are byte-identical to E1). Run one command at a time.

```powershell
# DirectML run per variant: stroke sweeps, corrective sequences, decoder time per point count.
# Add '--decoder-sessions', 'per-point-count' for one decoder session per point count, or
# '--pad-points', '32', '--pad-mode', 'label' (or 'repeat') for a fixed point count;
# '--profile-dir', "$e\e3\profiles\tiny" writes operator profiles for --profile-counts (1,5).
Set-Location D:\Projects\pu-erh_lab
$e = "build\tmp\explore"; $i = "$e\e4\input_4096"
.\scripts\ai_models\sam2\run_with_gpu_memory.ps1 `
  -Exe rust\puerh_mind\target\release\examples\segmentation_runtime_report.exe `
  -Arguments @('strokes', '--onnx-dir', "$e\e4\onnx_float32_batch16\tiny", '--device', 'directml',
               '--photos', "$i\photos", '--prompts', 'build\tmp\sm1\quality\prompts.json',
               '--targets', 'scripts\ai_models\sam2\click_targets.json',
               '--reference-masks', "$i\reference_masks",
               '--reference-scores', "$e\e1\compare\base_plus\compare.json",
               '--mask-dir', "$e\e3\masks\tiny", '--out', "$e\e3\runtime\tiny_directml.json") `
  -Out "$e\e3\runtime\tiny_directml_gpu_memory.json" -Log "$e\e3\runtime\tiny_directml.log"
```

```bash
E=build/tmp/explore; PY="$E/venv/Scripts/python -I"
# StrokePromptLogitsMatchPytorchWithinTolerance. --pad-points 32 --pad-mode label|repeat also
# decodes the padded prompts on both sides (the report runs it on one photo).
for v in tiny base_plus; do
  $PY scripts/ai_models/sam2/compare_stroke_outputs.py --repository $E/src/sam2 --weights $E/weights \
    --variant $v --onnx-dir $E/e4/onnx_float32_batch16/$v --photos $E/e4/input_4096/photos \
    --report $E/e3/runtime/${v}_directml.json --threads 4 --out $E/e3/compare/${v}_strokes.json
done
# Tables and sheets of the report. The sheets are in .gitignore (photos that must not be shared).
$PY scripts/ai_models/sam2/probe_strokes.py \
  --report "tiny=$E/e3/runtime/tiny_directml.json|$E/e3/runtime/tiny_sessions_directml.json|$E/e3/runtime/tiny_directml_gpu_memory.json" \
  --report "base plus=$E/e3/runtime/base_plus_directml.json|$E/e3/runtime/base_plus_sessions_directml.json|$E/e3/runtime/base_plus_directml_gpu_memory.json" \
  --out $E/e3/probe/final.json
$PY scripts/ai_models/sam2/make_stroke_sheets.py \
  --model "SAM 2.1 tiny=$E/e3/runtime/tiny_directml.json|$E/e3/masks/tiny" \
  --model "SAM 2.1 base plus=$E/e3/runtime/base_plus_directml.json|$E/e3/masks/base_plus" \
  --photos $E/e4/input_4096/photos --reference-masks $E/e4/input_4096/reference_masks \
  --out-dir docs/roadmap/alcedo_studio/edit/ai_mask_candidate/images/e3_strokes
```
