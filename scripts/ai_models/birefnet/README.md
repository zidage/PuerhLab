# BiRefNet export and measurement tools (AI Segmentation Mask, #458 step E2)

These scripts export the BiRefNet dichotomous segmentation and matting models to ONNX, check the
export against PyTorch, and make the evidence of the one-click Subject evaluation. The Rust program
`rust/puerh_mind/examples/segmentation_runtime_report.rs` (`matte` mode) measures the graphs with
the session settings of the `alcedo_mind` sidecar. The report is
[E2: BiRefNet one-click Subject and matting](../../../docs/roadmap/alcedo_studio/edit/ai_mask_candidate/e2_birefnet_subject_matting.md);
the plan is [AI Segmentation Mask master plan, section 8.16](../../../docs/roadmap/alcedo_studio/edit/ai_segmentation_mask_master_plan.md).

Weights, ONNX files, photos, and reports are not committed. Keep them under `build/tmp/explore/`.

## Sources

| Variant | Hugging Face repository | Revision | Backbone | Input |
| --- | --- | --- | --- | --- |
| `lite` | `ZhengPeng7/BiRefNet_lite` | `aa62cd87eafb9cc43056d08ef3615a14628b831d` | Swin-T | 1024 |
| `general` | `ZhengPeng7/BiRefNet` | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` | Swin-L | 1024 |
| `hr` | `ZhengPeng7/BiRefNet_HR` | `a7a562f6fd16021180f2f4348f4de003a2d3d1e1` | Swin-L | 2048 |
| `lite_matting` | `ZhengPeng7/BiRefNet_lite-matting` | `99c33412e3f58e1f33187abdc8c435c645243690` | Swin-T | 1024 |
| `matting` | `ZhengPeng7/BiRefNet-matting` | `eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9` | Swin-L | 1024 |
| `hr_matting` | `ZhengPeng7/BiRefNet_HR-matting` | `5d6b6f8adcb5b417c871b1d84ceaae9871355b7f` | Swin-L | 2048 |

- License: MIT (model cards and `LICENSE` of `https://github.com/ZhengPeng7/BiRefNet`).
- Each variant directory holds `birefnet.py`, `BiRefNet_config.py`, `config.json`, and
  `model.safetensors` of the revision above.
- `deform_conv2d_onnx_exporter` (`https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter`
  at `ed84d9114b4ab17c5e0a5dc8a412c0d57743a402`, MIT) is necessary only for
  `--deform-conv gathernd`.

## Files

| File | Purpose |
| --- | --- |
| `birefnet_graphs.py` | Variant table, model loading, the `SubjectMatte` wrapper, preprocessing, and the grid_sample form of `deform_conv2d`. |
| `export_onnx.py` | Export `subject_matte.onnx` and `export_report.json` for each variant. |
| `compare_onnx_outputs.py` | `OnnxMatteMatchesPytorchWithinTolerance`: PyTorch compared with ONNX Runtime CPU; writes the reference alpha for the Rust program and the alpha files for the probe. |
| `probe_mattes.py` | Foreground, soft share, edge width, parts, holes, islands, subject recall, and region metrics. |
| `make_matte_sheets.py` | Overview, per-photo, and detail-crop sheets of the report. |
| `convert_fp16.py` | Float16 copy of a graph (float32 inputs and outputs; GridSample and Range stay float32). |

`scripts/ai_models/sam2/run_with_gpu_memory.ps1` records the GPU memory.

## Graph

| File | Inputs | Outputs |
| --- | --- | --- |
| `subject_matte.onnx` | `pixel_values` [1, 3, S, S] | `alpha` [1, 1, S, S] |

Data rules:

- S is 1024 or 2048 (see the table). The photo is stretched to S x S without padding. Pixel values
  are `(value / 255 - mean) / std` with the ImageNet mean `(0.485, 0.456, 0.406)` and std
  `(0.229, 0.224, 0.225)`. `export_report.json` states these values in `model_input`.
- `alpha` is the sigmoid of the last BiRefNet output, in [0, 1]. The host resizes it to the photo
  size with bilinear interpolation.
- `deform_conv2d` is exported as one GridSample and one 1 x 1 Conv per kernel point
  (`--deform-conv grid_sample`, the default). The decomposition of the BiRefNet ONNX tutorial
  (`--deform-conv gathernd`) makes tensors of up to 822 MB (lite) and DirectML stops with
  `E_OUTOFMEMORY` (`8007000E`).

## Commands

```bash
# Python environment and sources (Git Bash, Windows)
E=build/tmp/explore
python -m venv $E/venv_birefnet
$E/venv_birefnet/Scripts/python -m pip install -r scripts/ai_models/birefnet/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
W=$E/weights/birefnet
declare -A REV=([BiRefNet_lite]=aa62cd87eafb9cc43056d08ef3615a14628b831d
  [BiRefNet]=e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4
  [BiRefNet_HR]=a7a562f6fd16021180f2f4348f4de003a2d3d1e1
  [BiRefNet_lite-matting]=99c33412e3f58e1f33187abdc8c435c645243690
  [BiRefNet-matting]=eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9
  [BiRefNet_HR-matting]=5d6b6f8adcb5b417c871b1d84ceaae9871355b7f)
for m in "${!REV[@]}"; do
  mkdir -p $W/$m
  for f in birefnet.py BiRefNet_config.py config.json model.safetensors; do
    curl -sSL -o $W/$m/$f "https://huggingface.co/ZhengPeng7/$m/resolve/${REV[$m]}/$f"
  done
done
# revisions.json: {"ZhengPeng7/<name>": "<revision>", ...} with the values above.

PY="$E/venv_birefnet/Scripts/python -I"
Q=build/tmp/sm1/quality
$PY scripts/ai_models/birefnet/export_onnx.py --weights $W --revisions $W/revisions.json \
  --out-root $E/e2/onnx
for v in lite lite_matting general matting hr hr_matting; do
  $PY scripts/ai_models/birefnet/compare_onnx_outputs.py --weights $W --variant $v \
    --onnx-dir $E/e2/onnx/$v --photos $Q/photos --prompts $Q/prompts.json \
    --out $E/e2/compare/$v/compare.json
  $PY scripts/ai_models/birefnet/probe_mattes.py --alpha-dir $E/e2/compare/$v/alpha \
    --prompts $Q/prompts.json --subject-masks $E/e1/compare/base_plus/reference_masks \
    --label $v --out $E/e2/probe/$v.json \
    --region "DSC00674.jpg:hair:0.15,0.0,0.95,0.45" --region "P1024439.jpg:hair:0.66,0.27,0.84,0.5" \
    --region "DSC_1535.jpg:hair:0.38,0.0,0.64,0.16" --region "DSC00326.jpg:hair:0.44,0.6,0.6,0.76" \
    --region "DSC_1188.jpg:fur:0.3,0.1,0.72,0.64" --region "DSC00936.jpg:person:0.47,0.63,0.68,0.93"
done
```

```powershell
# DirectML measurement with GPU memory (PowerShell, Windows). Repeat for each variant. Add
# '--reference-alpha', "$e\compare\lite\reference_alpha" for RustPipelineMatteMatchesPytorchReference.
Set-Location D:\Projects\pu-erh_lab
$e = "build\tmp\explore\e2"; $q = "build\tmp\sm1\quality"
.\scripts\ai_models\sam2\run_with_gpu_memory.ps1 `
  -Exe rust\puerh_mind\target\release\examples\segmentation_runtime_report.exe `
  -Arguments @('matte', '--onnx-dir', "$e\onnx\lite", '--device', 'directml',
               '--photos', "$q\photos", '--prompts', "$q\prompts.json",
               '--overlay-dir', "$e\overlays\lite", '--overlay-max-edge', '2048',
               '--out', "$e\runtime\lite_directml.json") `
  -Out "$e\runtime\lite_directml_gpu_memory.json" -Log "$e\runtime\lite_directml.log"
# Operator placement: a separate run with '--profile-dir', "$e\profiles\lite".
```

```bash
# Comparison sheets of the report. The output directory is in .gitignore: the images show photos
# that must not be shared again, so they are not distributed with the GitHub repository.
O=$E/e2/overlays
$PY scripts/ai_models/birefnet/make_matte_sheets.py \
  --model "lite=$O/lite" --model "lite matting=$O/lite_matting" --model "general=$O/general" \
  --model "matting=$O/matting" --model "HR=$O/hr" --model "HR matting=$O/hr_matting" \
  --reference "SAM 2.1 base+ box=$E/e1/overlays/base_plus" \
  --photos $Q/photos --prompts $Q/prompts.json --quality 72 \
  --out-dir docs/roadmap/alcedo_studio/edit/ai_mask_candidate/images/e2_birefnet \
  --crop "DSC00674.jpg:hair:0.15,0.0,0.95,0.45" --crop "P1024439.jpg:hair:0.66,0.27,0.84,0.5" \
  --crop "DSC_1535.jpg:hair:0.38,0.0,0.64,0.16" --crop "DSC00326.jpg:hair:0.44,0.6,0.6,0.76" \
  --crop "DSC_1188.jpg:fur:0.3,0.1,0.72,0.64" --crop "DSC00936.jpg:person:0.47,0.63,0.68,0.93"
```
