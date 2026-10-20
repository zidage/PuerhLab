# E2: BiRefNet one-click Subject and matting

**Issue:** #458, step E2. **Plan:** [AI Segmentation Mask master plan, section 8.16](../ai_segmentation_mask_master_plan.md#816-grid-holes-and-model-re-evaluation).
**Date:** 2026-10-19. **Branch:** `feature/ai-segmentation-sm1-quality-review`.
**Status:** measurements stopped by the user on 2026-10-19. User assessment: the latency is not
usable for the product. The float16 measurement of lite matting, matting, HR, and HR matting was
not completed (section 6.3). User decision (2026-10-19): only BiRefNet lite stays a candidate; the
other five variants are dropped.

> **Images are local only.** The images under `images/` show photos that must not be shared
> again. They are in `.gitignore` and are not distributed with the GitHub repository. On a
> checkout without them, run the sheet command of the
> [script README](../../../../../scripts/ai_models/birefnet/README.md#commands) with the local
> evidence under `build/tmp/explore/e2/`.

## 1. Question

Plan section 8.16.3 found that other photo applications use a one-click Subject mask as the main
entry. Step E2 answers: can one dichotomous segmentation or matting model supply the one-click
Subject mask and the hair edge without a separate matte step? The step also records the latency,
the memory, the file size, and the photos with more than one subject.

## 2. Result summary

- **Subject mask: BiRefNet lite finds the expected subject on 12 of 13 photos that have one.** It
  finds the person, the animal, the car, the tram, the balloon (0.4 percent of the frame), and the
  paddle boarder (1.3 percent) without a prompt. The general (Swin-L) and HR variants give almost
  the same masks. On the duck
  photo (`DSC00562`), all variants select the wall with the ducks on it, not the ducks. On the
  lupine photo, lite and general select almost nothing, which is correct for a photo without a
  main subject.
- **Several subjects:** BiRefNet selects all salient subjects in one mask (both musicians on
  `P1024439`, three people on `DSC00527`). A user who wants one of them needs the click selection
  of step E4.
- **Hair edge:** the lite and general masks follow hair and fur closely (curly hair, backlit
  ponytail, squirrel tail) with a narrow soft edge (1 to 7 model pixels). Single strands are mostly
  in or out, not semi-transparent. The matting variants give soft strands (soft share in the
  squirrel fur crop 0.089 to 0.106; lite 0.025).
- **Matting variants miss rigid objects.** Lite matting misses the car (`DSC_1456`), the tram, and
  the balloon; matting misses the car and the balloon. They are not usable as the only Subject
  model.
- **Export:** the BiRefNet ONNX tutorial method (GatherND decomposition of `deform_conv2d`) makes
  tensors of up to 822 MB; DirectML stops with `E_OUTOFMEMORY`. A GridSample decomposition of the
  same operation (maximum relative difference to torchvision 3.6e-6) runs completely on DirectML.
  All exported graphs agree with PyTorch (14 of 14 photos per variant, minimum IoU 0.9998).
- **Cost on the RTX 3080 Laptop GPU (8 GB):**
  - lite and lite matting (float32): 218 ms warm p50, 5.7 GB peak dedicated GPU memory, 183 MB
    file. SAM 2.1 tiny (E1): 86 ms (encoder and one decoder run), 1.36 GB.
  - lite (float16): 175 ms warm p50, 3.2 GB, 96 MB file.
  - general and matting (Swin-L, 1024, float32): 3.3 to 15 s, because the graph needs more than
    8 GB and DirectML uses up to 6.2 GB of shared system memory. 932 MB file. General in float16:
    351 ms, 5.7 GB, 472 MB file (measured while a CPU run was active).
  - HR and HR matting (Swin-L, 2048): DirectML fails (`887A0001`). CPU: 34 s.
  - CPU: lite 3.2 s, Swin-L 8.5 s.
  - DirectML stopped intermittently with `887A0006` (`DXGI_ERROR_DEVICE_HUNG`) in 4 runs, with
    float32 and float16 graphs. A repeat of the same run passed in 3 cases; the lite matting
    float16 run was not repeated.
- **License:** MIT for the code and the weights. Some training sets of the matting variants are
  research data sets; a legal check is necessary before shipping (section 7).
- **Answer to the question: no single model.** BiRefNet lite gives a good one-click Subject mask,
  and its edge is good on hair at the mask level, but it does not give a soft matte on single
  strands. The matting variants give the soft matte, but only on people and animals.
- **User assessment (2026-10-19):** the measured speed is not usable for the product.
- **User decision (2026-10-19):** only BiRefNet lite stays a candidate for the Subject task. lite
  matting, general, matting, HR, and HR matting are dropped. The quality of lite on this set is
  the reference for the Subject task in step E6.

## 3. Method

### 3.1 Inputs

| Item | Value |
| --- | --- |
| Photos | The SM1 quality set (`build/tmp/sm1/quality/`): 14 photos (plan section 8.15). The model has no prompt; the box prompt of each photo names the expected subject for the recall metric. |
| Variants | `lite` (Swin-T, 1024), `general` (Swin-L, 1024), `hr` (Swin-L, 2048), `lite_matting` (Swin-T, 1024), `matting` (Swin-L, 1024), `hr_matting` (Swin-L, 2048). The Hugging Face repositories and revisions are in the [script README](../../../../../scripts/ai_models/birefnet/README.md#sources). |
| Not tested | `BiRefNet_lite-2K` (2560 x 1440 input, landscape only), `BiRefNet_dynamic`, `BiRefNet-portrait`, and the DIS, HRSOD, and COD benchmark variants. |
| Reference | The SAM 2.1 base plus box masks of step E1 (`build/tmp/explore/e1/compare/base_plus/reference_masks/`). |
| Code | Hugging Face `birefnet.py` of each variant (the BiRefNet repository code with the backbone of the variant); `ZhengPeng7/BiRefNet` at `97e4f35d74043427b0e9ba935ca10991853a36bc` for the license and the ONNX tutorial. |
| Scripts | [`scripts/ai_models/birefnet/`](../../../../../scripts/ai_models/birefnet/README.md) and `rust/puerh_mind/examples/segmentation_runtime_report.rs` (`matte` mode) |
| Evidence | `build/tmp/explore/e2/` on the Windows machine (not committed): `onnx/`, `onnx_gathernd/`, `onnx_fp16/`, `compare/`, `probe/`, `runtime/`, `runtime_gathernd/`, `profiles/`, `overlays/` |

### 3.2 Machine

The machine of step E1: Windows 11 build 22635, Intel Core i7-12700H, 31.8 GB, NVIDIA GeForce RTX
3080 Laptop GPU 8 GB, driver 610.62. ONNX Runtime 1.24 through `ort` 2.0.0-rc.12 (Rust program,
production session settings of `ort_runtime::load_session`). Python 3.14.2, PyTorch 2.14.1 (CPU),
transformers 5.20.2, onnxruntime 1.31.0. macOS Core ML is not measured.

### 3.3 Export

One graph per variant: `subject_matte.onnx`, `pixel_values` [1, 3, S, S] to `alpha` [1, 1, S, S]
(sigmoid in the graph), opset 17, TorchScript exporter. The input is the whole photo stretched to
S x S with the ImageNet normalization, the preprocessing of the model cards. The Rust program
resizes `alpha` to the photo size with bilinear interpolation.

BiRefNet uses `torchvision.ops.deform_conv2d` in the ASPP blocks of the decoder (kernels 1 x 1,
3 x 3, and 7 x 7). The TorchScript exporter has no symbolic for it:

- **GatherND decomposition** (`deform_conv2d_onnx_exporter`, the method of the BiRefNet ONNX
  tutorial): it gathers the four bilinear corners of all kernel points at once. In lite, the 7 x 7
  kernel at 256 x 256 with 64 channels makes 21 tensors of 822 MB; the intermediate tensors of the
  graph add up to 35.2 GB. On DirectML the first inference stops with `8007000E`
  (`E_OUTOFMEMORY`) after 7.1 GB dedicated and 16.3 GB shared GPU memory. Two exporter fixes for
  PyTorch 2.14 were necessary (the `JitScalarType` module and the tensor sizes from strides).
- **GridSample decomposition** (`birefnet_graphs.deform_conv2d_grid_sample`, used for all
  results): one bilinear `grid_sample` (zero padding, `align_corners=True`) and one 1 x 1
  convolution per kernel point, accumulated. The sampling positions are the torchvision positions.
  `GridSampleDeformConvMatchesTorchvision`: maximum relative difference 3.6e-6 on random inputs
  for the kernel sizes 1, 3, and 7 (required 1e-5; torchvision itself differs from a float64
  computation by the same order). The largest tensor of lite is 50 MB (Swin-T MLP); the
  intermediate tensors add up to 3.5 GB.

### 3.4 Metrics

- `OnnxMatteMatchesPytorchWithinTolerance`: the PyTorch model (with torchvision `deform_conv2d`)
  compared with the ONNX Runtime CPU graph on the same pixels; IoU of alpha > 0.5 at the photo
  size, required 0.99, and the maximum absolute alpha difference.
- `RustPipelineMatteMatchesPytorchReference`: the alpha of the Rust program (Rust resize, DirectML
  or CPU) compared with the PyTorch alpha at the photo size.
- Probe metrics (`probe_mattes.py`) on the ONNX Runtime CPU alpha at the model size:
  - **foreground:** share of pixels with alpha > 0.5;
  - **edge width:** pixels with 0.05 < alpha < 0.95 divided by the boundary pixels of the
    alpha > 0.5 mask, in model pixels. A hard mask gives 1 to 2;
  - **parts:** connected parts with at least 1 percent of the foreground area;
  - **subject recall:** share of the SAM 2.1 base plus box mask of the photo inside the foreground;
  - **outside box:** share of the foreground outside the box of that prompt (other subjects or
    background);
  - **soft share in a crop:** share of soft pixels (0.05 < alpha < 0.95) in the hair and fur crops
    of section 5.4.
- Latency and memory: `segmentation_runtime_report matte` with the production session settings;
  load, first inference, 20 warm runs on the first photo, one run per other photo. GPU memory from
  `run_with_gpu_memory.ps1` (Windows `GPU Process Memory` counters, every 250 ms). Operator
  placement from a separate run with profiling (2 inferences). CPU: one photo (`DSC_1535`), 5 warm
  runs (3 for HR).

## 4. Export and agreement

| Variant | Parameters | Weights bytes (`model.safetensors`) | `subject_matte.onnx` bytes | Export time | Agreement (passed / photos) | Minimum IoU | Maximum \|alpha difference\| |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| lite | 44.3M | 177,634,392 (float32) | 183,439,077 | 26 s | 14 / 14 | 0.99997 | 0.0004 |
| lite matting | 44.3M | 88,965,896 (float16) | 183,439,077 | 26 s | 14 / 14 | 0.99980 | 0.0001 |
| general | 220.2M | 444,473,596 (float16) | 932,186,731 | 55 s | 14 / 14 | 1.00000 | 0.0007 |
| matting | 220.2M | 884,878,856 (float32) | 932,186,731 | 64 s | 14 / 14 | 0.99983 | 0.0001 |
| HR | 220.2M | 444,473,596 (float16) | 932,192,968 | 129 s | 14 / 14 | 0.99999 | 0.0002 |
| HR matting | 220.2M | 444,473,596 (float16) | 932,192,875 | 107 s | 14 / 14 | 0.99995 | 0.0001 |

The graphs are float32 (the half-precision weights are converted at load). All six graphs have
23,954 (Swin-T) or 29,522 (Swin-L) nodes, 300 of them GridSample.
`MatteOutputShapeMatchesExportedSpecification` passed for all variants that ran.
`SegmentationGraphsLoadWithProductionSessionSettings` passed on `directml` and `cpu` for all six.

**Rust pipeline compared with PyTorch** (`RustPipelineMatteMatchesPytorchReference`; the four
1024 variants on DirectML, HR and HR matting on the CPU). The Rust program resizes with
`image::imageops` (Triangle); PIL uses its own bilinear filter. The mean absolute alpha difference is
at most 0.0075 on every photo. The IoU of alpha > 0.5 is at least 0.99 on 14 of 14 photos for
general, 13 for lite and HR, 10 for lite matting, 9 for matting and HR matting. All values below
0.99 are on photos where the variant is uncertain: a small or failed foreground (lite on the lupine
photo: 0.716 at 0.2 percent foreground; matting on `DSC_1456`: 0.381; lite matting on the tram:
0.797) or the soft second musician of HR on `P1024439` (0.923). There, a small resize difference
moves many pixels across 0.5. Result as in E1: the product must use one resize implementation on
all platforms.

SHA-256 of the ONNX files:

| File | SHA-256 |
| --- | --- |
| lite `subject_matte.onnx` | `dcb89537a5925acbd6f6601e239e0ab573fb1f7cf13d8b287aa42cfcf737a5ef` |
| lite matting `subject_matte.onnx` | `18f0949a52d3d289a6bf48e7befa697bc1bfe6640aefcf1e777b06149c3312ca` |
| general `subject_matte.onnx` | `33fdf06b723963d8800a65619377fbdff4dd4a902cef66e84c249fc610f23abb` |
| matting `subject_matte.onnx` | `25b0c7b5e4c0ad6b8db39e9571d379242edafafd24f567b78dd3e69a50b106ec` |
| HR `subject_matte.onnx` | `b7bacd9c202a14c7a6df0f34b0f123a6c6fe8836132b8374b29bf19894d036e1` |
| HR matting `subject_matte.onnx` | `058e4e48d37dd81516d00959f3a94ad81d839ae06a7de5f05e622b043d01275d` |
| BiRefNet `LICENSE` (MIT) | `92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb` |

## 5. Quality

### 5.1 Overview

One row per photo, one column per variant; the last column is the SAM 2.1 base plus box mask of
step E1 with its prompt. The overlay blends the alpha in blue (soft edges stay soft). The HR
overlays are from the CPU, because the HR graphs fail on DirectML.

![Subject mattes on the quality set](images/e2_birefnet/overview.jpg)

### 5.2 Probe metrics

Foreground share / subject recall / edge width in model pixels / parts:

| Photo | lite | lite matting | general | matting | HR | HR matting |
| --- | --- | --- | --- | --- | --- | --- |
| `DSC_1535` | 0.222 / 1.00 / 1.5 / 1 | 0.222 / 1.00 / 2.2 / 1 | 0.223 / 1.00 / 1.5 / 1 | 0.222 / 1.00 / 2.0 / 1 | 0.223 / 1.00 / 1.9 / 1 | 0.222 / 1.00 / 2.8 / 1 |
| `DSC00326` | 0.026 / 0.99 / 1.2 / 1 | 0.025 / 0.97 / 3.4 / 1 | 0.026 / 0.99 / 1.5 / 1 | 0.026 / 0.98 / 2.8 / 1 | 0.026 / 0.99 / 2.2 / 1 | 0.026 / 0.99 / 4.2 / 1 |
| `DSC00674` | 0.679 / 1.00 / 6.7 / 1 | 0.693 / 1.00 / 27.8 / 1 | 0.678 / 1.00 / 4.6 / 1 | 0.679 / 1.00 / 10.4 / 1 | 0.678 / 1.00 / 10.2 / 1 | 0.678 / 1.00 / 16.1 / 1 |
| `DSC00936` | 0.022 / 1.00 / 1.0 / 1 | 0.022 / 0.99 / 2.0 / 1 | 0.022 / 1.00 / 1.3 / 1 | 0.022 / 0.99 / 1.9 / 1 | 0.022 / 1.00 / 1.7 / 1 | 0.022 / 0.99 / 3.0 / 1 |
| `P1024439` | 0.262 / 1.00 / 1.7 / 3 | 0.247 / 0.99 / 3.9 / 2 | 0.244 / 1.00 / 1.7 / 2 | 0.250 / 1.00 / 4.2 / 2 | 0.102 / 0.99 / 28.3 / 6 | 0.249 / 1.00 / 6.9 / 2 |
| `DSC_1188` | 0.110 / 1.00 / 2.2 / 1 | 0.102 / 0.96 / 9.4 / 1 | 0.112 / 1.00 / 1.7 / 1 | 0.105 / 0.98 / 7.6 / 1 | 0.111 / 1.00 / 4.9 / 1 | 0.106 / 0.98 / 11.3 / 1 |
| `DSC00527` | 0.024 / 1.00 / 0.9 / 2 | 0.024 / 0.99 / 3.3 / 2 | 0.023 / 1.00 / 1.3 / 2 | 0.014 / 0.17 / 6.3 / 2 | 0.023 / 0.99 / 1.8 / 3 | 0.018 / 0.99 / 6.8 / 4 |
| `2088064441` | 0.005 / 0.98 / 1.2 / 1 | 0.006 / 0.92 / 15.3 / 3 | 0.004 / 0.98 / 1.4 / 1 | 0.001 / 0.26 / 10.5 / 4 | 0.004 / 0.92 / 1.8 / 1 | 0.001 / 0.10 / 12.1 / 8 |
| `DSC00728` | 0.003 / 0.99 / 0.8 / 1 | 0.000 / 0.00 / 0.0 / 0 | 0.003 / 0.99 / 1.0 / 1 | 0.000 / 0.07 / 7.0 / 8 | 0.003 / 0.99 / 1.2 / 1 | 0.000 / 0.01 / 4.7 / 1 |
| `5761824892` | 0.241 / 1.00 / 1.4 / 1 | 0.230 / 0.79 / 23.4 / 6 | 0.241 / 1.00 / 1.5 / 1 | 0.220 / 0.91 / 8.1 / 1 | 0.241 / 1.00 / 2.4 / 1 | 0.226 / 0.94 / 11.6 / 1 |
| `DSC00562` | 0.372 / 0.98 / 1.9 / 1 | 0.373 / 0.98 / 16.7 / 1 | 0.365 / 0.01 / 1.6 / 1 | 0.072 / 0.96 / 42.3 / 6 | 0.368 / 0.63 / 7.7 / 1 | 0.045 / 0.98 / 50.3 / 4 |
| `DSC00602` | 0.274 / 1.00 / 1.6 / 1 | 0.000 / 0.00 / 543.7 / 1 | 0.273 / 1.00 / 1.5 / 1 | 0.268 / 0.99 / 12.5 / 1 | 0.273 / 1.00 / 3.7 / 1 | 0.272 / 1.00 / 16.8 / 1 |
| `DSC_1456` | 0.065 / 1.00 / 1.3 / 1 | 0.000 / 0.00 / 52.6 / 3 | 0.065 / 1.00 / 1.5 / 1 | 0.010 / 0.15 / 13.6 / 3 | 0.065 / 1.00 / 2.3 / 1 | 0.000 / 0.00 / 0.0 / 0 |
| `Nikon-D800e-raw-00007` | 0.003 / 0.00 / 3.3 / 1 | 0.000 / 0.00 / 283.9 / 5 | 0.000 / 0.00 / 0.0 / 0 | 0.000 / 0.00 / 0.0 / 0 | 0.019 / 1.00 / 8.4 / 1 | 0.001 / 0.07 / 78.2 / 1 |

- **Subject recall** is at least 0.92 for lite, general, and HR on every photo with a clear
  subject. The exceptions are the photos where the model selects a different subject than the box
  prompt: `DSC00562` (the wall with the ducks; general and HR include only part of the duck in the
  box) and the lupine photo (no subject; HR selects the rock of the box prompt).
- **Edge width** of lite and general is 1 to 2 model pixels on most photos (a hard mask) and 4.6
  to 6.7 on the `DSC00674` close-up (hair). The matting variants have wider soft edges (2 to 28).
  Large values on photos without a subject (543.7 for lite matting on the tram) come from an alpha
  that is soft everywhere and near zero.
- **Several subjects:** `P1024439` has 2 to 3 parts (both musicians, lite also the microphone
  stand) and 74 to 75 percent of the foreground outside the bassist box; HR selects mostly the
  bassist (39 percent outside, 6 small parts). `DSC00527` has 2 to 3 parts (three people).

### 5.3 Result per photo

| Case | Photo | lite | general | HR | lite matting | matting |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `DSC_1535` | Person, curly hair edge clean | Same | Same | Same, softer hair | Same, softer hair |
| 2 | `DSC00326` | Person from behind, long hair | Same | Same | Same, soft hair | Same, soft hair |
| 7 | `DSC00674` | Whole close-up person | Same | Same | Same, wide soft edge | Same |
| 10 | `DSC00936` | Person against mist | Same | Same | Same | Same |
| extra | `P1024439` | Both musicians and the microphone stand | Both musicians; a bright tent patch at the ponytail | Mostly the bassist | Both musicians | Both musicians |
| 3 | `DSC_1188` | Squirrel with tail | Same | Same | Squirrel, soft tail | Squirrel, soft tail |
| 4 | `DSC00527` | Three people (not the sky) | Same | Same | Same | Two people, not the kneeling woman |
| 5 | `2088064441` | Paddle boarder with board | Same | Same | Boarder and parts of the water | Partly |
| 6 | `DSC00728` | Balloon | Same | Same | Nothing | Nothing (fragments) |
| 8 | `5761824892` | Car with kayak | Same | Same | Car and a tree branch | Car with kayak |
| 11 | `DSC00562` | The wall with the ducks | The wall | The wall | The wall | Part of the wall |
| 12 | `DSC00602` | Tram | Same | Same | Nothing | Tram |
| extra | `DSC_1456` | Car | Same | Same | Nothing | Nothing |
| extra | Nikon lupine | Almost nothing (0.3 percent) | Nothing | The rock | Fragments | Nothing |

![DSC_1535](images/e2_birefnet/DSC_1535.jpg)

![DSC00326](images/e2_birefnet/DSC00326.jpg)

![DSC00674](images/e2_birefnet/DSC00674.jpg)

![DSC00936](images/e2_birefnet/DSC00936.jpg)

![P1024439](images/e2_birefnet/P1024439.jpg)

![DSC_1188](images/e2_birefnet/DSC_1188.jpg)

![DSC00527](images/e2_birefnet/DSC00527.jpg)

![2088064441](images/e2_birefnet/2088064441.jpg)

![DSC00728](images/e2_birefnet/DSC00728.jpg)

![5761824892](images/e2_birefnet/5761824892.jpg)

![DSC00562](images/e2_birefnet/DSC00562.jpg)

![DSC00602](images/e2_birefnet/DSC00602.jpg)

![DSC_1456](images/e2_birefnet/DSC_1456.jpg)

![Nikon-D800e-raw-00007](images/e2_birefnet/Nikon-D800e-raw-00007.jpg)

### 5.4 Hair and fur edges

Crops of the 2048 px alpha (left) and of the photo composited on a flat green background with that
alpha (right), so a soft strand shows the background through it:

![Hair and fur alpha](images/e2_birefnet/hair_detail_alpha.jpg)

![Hair and fur composite](images/e2_birefnet/hair_detail_composite.jpg)

Soft share in the crops (share of pixels with 0.05 < alpha < 0.95):

| Crop | lite | lite matting | general | matting | HR | HR matting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DSC00674` hair | 0.040 | 0.055 | 0.023 | 0.049 | 0.027 | 0.046 |
| `P1024439` hair | 0.031 | 0.042 | 0.022 | 0.053 | 0.066 | 0.045 |
| `DSC_1535` hair | 0.032 | 0.047 | 0.030 | 0.045 | 0.022 | 0.037 |
| `DSC00326` hair | 0.018 | 0.046 | 0.022 | 0.036 | 0.016 | 0.031 |
| `DSC_1188` fur | 0.025 | 0.106 | 0.017 | 0.089 | 0.032 | 0.077 |
| `DSC00936` person | 0.017 | 0.035 | 0.022 | 0.034 | 0.015 | 0.026 |

- lite and general: the outline follows the curls, the ponytail, and the fur tips; the strands are
  mostly in or out. The backlit flyaways of `P1024439` are partly cut. General adds a bright tent
  patch next to the ponytail.
- lite matting and matting: the squirrel tail and the backlit hair are semi-transparent; the
  background shows through the strands. The `DSC00674` close-up gets a wide soft band on the dark
  hair against the sunset (edge width 28 for lite matting), which looks blurred.
- HR: the edge is sharper than general at 2048, but HR does not add soft strands.

## 6. Performance

### 6.1 DirectML (RTX 3080 Laptop GPU)

Milliseconds. Warm values are p50 / p95 / maximum (20 runs on the first photo).

| Variant | Load | First run | Warm | Other photos, p50 / maximum | Peak dedicated GPU memory | Peak shared GPU memory | Private bytes | Peak working set |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| lite | 5,883 | 1,329 | 218.2 / 222.4 / 222.5 | 217.6 / 340.2 | 5,721 MB | 630 MB | 6,739 MB | 2,920 MB |
| lite matting | 5,157 | 485 | 218.3 / 221.3 / 221.3 | 221.5 / 650.0 | 5,721 MB | 630 MB | 6,716 MB | 1,384 MB |
| general | 7,548 | 8,303 | 15,153 / 16,319 / 18,685 | 15,291 / 19,911 | 6,594 MB | 6,150 MB | 12,502 MB | 11,348 MB |
| matting | 6,979 | 6,063 | 11,707 / 15,929 / 20,111 | 3,921 / 4,990 | 7,070 MB | 5,994 MB | 12,491 MB | 11,789 MB |
| HR | 9,589 | fails: `887A0001` (`DXGI_ERROR_INVALID_CALL`) in the first fused node | — | — | 5,355 MB before the failure | 5,867 MB | — | — |
| HR matting | — | fails: same error | — | — | 5,869 MB before the failure | 6,939 MB | — | — |
| lite, float16 | 7,636 | 543 | 174.5 / 182.0 / 187.5 | 190.7 / — | 3,226 MB | 342 MB | — | — |
| general, float16 (a CPU run was active) | 20,531 | 796 | 350.5 / 423.1 / 462.3 | — | 5,663 MB | 348 MB | — | — |
| lite matting, float16 | — | fails: `887A0006` (`DXGI_ERROR_DEVICE_HUNG`); not repeated | — | — | 3,226 MB | 426 MB | — | — |
| *SAM 2.1 tiny (E1), encoder + decoder* | *1,471 + 185* | *116* | *75.9 + 10.4* | — | *1,355 MB* | *237 MB* | *1,807 MB* | *789 MB* |

**Operator placement (DirectML, production settings).** All nodes run on DirectML (601 fused
nodes, 0 CPU nodes) in the four variants that run. Profiled kernel time per inference: lite
266 ms, lite matting 278 ms, general 743 ms, matting 778 ms. The Swin-L kernel time is 0.75 s,
but the wall time is 3 to 17 s and changes between runs and processes (matting: 3.9 s per photo,
11.7 s on the warm photo; a separate profiling process: 3.3 s): the graph does not fit in the
8 GB of dedicated memory, and DirectML pages tensors through shared system memory.

The float16 lite graph places 724 nodes on the CPU provider (2,989 on DirectML; the casts around
the float32 GridSample and Range nodes), 22 ms of the 714 ms profiled kernel time of 2 runs.

### 6.2 CPU (i7-12700H)

One photo (`DSC_1535`), production session settings.

| Variant | Warm p50 / p95 | Peak working set |
| --- | --- | ---: |
| lite | 3,221 / 3,369 | 1,519 MB |
| lite matting | 3,181 / 3,196 | 1,519 MB |
| general | 8,489 / 8,498 | 3,573 MB |
| matting | 8,319 / 8,999 | 3,573 MB |
| HR | 33,822 / 33,910 | 11,014 MB |
| HR matting | 34,511 / 37,820 | 11,014 MB |
| *SAM 2.1 tiny (E1), encoder* | *1,118 / 1,156* | *835 MB* |

### 6.3 Float16 graphs and the stop of the measurement

- **Float16 graph** (`convert_fp16.py`: float16 inside the graph, float32 inputs and outputs,
  GridSample and Range kept in float32; the converter made 295 duplicate Cast nodes, which the
  script removes). Lite: 175 ms and 3.2 GB (float32: 218 ms and 5.7 GB). General: 351 ms and
  5.7 GB without paging (float32: 15 s). The float16 alpha agrees with the float32 PyTorch alpha:
  IoU of at least 0.984 on 13 of 14 photos for lite (lupine photo 0.714, as in float32) and at
  least 0.991 on 14 of 14 for general.
- The first float16 runs of lite and general and the float16 run of lite matting stopped with
  `887A0006`; one float32 lite run also stopped with it. The repeats of lite and general passed.
  The cause is not known.
- **Stopped:** the user stopped the measurement before the float16 runs of lite matting, matting,
  HR, and HR matting were complete. User assessment: the measured speed is not usable.
- **Not tried:** fewer graph partitions (the 601 DirectML fused nodes come from the 300 GridSample
  nodes), constant folding of the sampling grids before the export, a smaller input (768 px), and a
  tiled decoder.

## 7. License

| Item | License | Commercial use |
| --- | --- | --- |
| BiRefNet code (`ZhengPeng7/BiRefNet`, `birefnet.py` in the Hugging Face repositories) | MIT (`LICENSE`, 1,066 bytes) | yes |
| BiRefNet weights (all six variants) | MIT (model cards: `license: mit`) | yes |
| `deform_conv2d_onnx_exporter` (only for the GatherND export, not shipped) | MIT | not shipped |
| Training data of the matting variants (model cards) | P3M-10k, TR-humans, AM-2k, AIM-500, Human-2k, Distinctions-646, HIM2K, PPM-100. Some of these data sets are published for research use. | The MIT license of the weights is not limited by the data set terms, but a legal check of this point is necessary before shipping. |

A published model directory needs the MIT text. The Meta SAM License is not necessary.

## 8. Result per variant

User assessment (2026-10-19): the measured speed is not usable for the product. User decision
(2026-10-19): only lite stays a candidate; the other variants are dropped.

| Variant | Quality on the set | DirectML warm p50 | File | Peak dedicated GPU memory | Result |
| --- | --- | ---: | ---: | ---: | --- |
| lite | Expected subject on 12 of 13 photos with a subject; hair edge clean but hard; all salient subjects in one mask | 218 ms (float16: 175 ms) | 183 MB (float16: 96 MB) | 5.7 GB (float16: 3.2 GB) | **Kept as the only candidate** (user decision); latency not usable in the measured form |
| lite matting | Soft hair and fur; misses the car, the tram, and the balloon | 218 ms | 183 MB | 5.7 GB | Dropped: misses rigid objects |
| general | Same as lite; a tent patch at the ponytail | 15 s (float16: 351 ms) | 932 MB (float16: 472 MB) | > 8 GB (float16: 5.7 GB) | Dropped: no visible gain over lite |
| matting | Soft hair; misses the car and the balloon | 3.9 to 12 s | 932 MB | > 8 GB | Dropped |
| HR | Sharpest outline; mostly one musician on `P1024439` | fails | 932 MB | > 8 GB | Dropped: not affordable (CPU 34 s) |
| HR matting | Soft hair; misses the car, the balloon, and the boarder | fails | 932 MB | > 8 GB | Dropped: not affordable (CPU 35 s) |

## 9. Open items and next steps

- **Subject task.** BiRefNet lite is the only kept candidate, but its measured speed is not usable
  (user). Step E6 compares it with faster Subject candidates; its quality on this set is the
  reference.
- **Edge stage (step E5).** No variant gives both the Subject mask on all subjects and soft
  strands. E5 is still open (guided-filter upsampling of a soft matte).
- **Several subjects.** One Subject mask holds all salient subjects. The selection of one subject
  is step E4 (click selection with candidates).
- **Confidence check.** On the lupine photo lite returns 0.3 percent of the frame. A minimum
  foreground share or a minimum alpha could hide such a mask (plan section 8.16.3).
- **Aspect ratio.** The input is stretched to a square. `BiRefNet_lite-2K` (2560 x 1440) is not
  tested.
- **macOS Core ML** is not measured. GridSample support of Core ML is not known.

## 10. Changed files

- `scripts/ai_models/birefnet/`: `birefnet_graphs.py`, `export_onnx.py`,
  `compare_onnx_outputs.py`, `probe_mattes.py`, `make_matte_sheets.py`, `convert_fp16.py`,
  `requirements.txt`, `README.md`.
- `rust/puerh_mind/examples/segmentation_runtime_report.rs`: new `matte` mode (one graph from
  pixels to alpha, soft overlays and alpha images, comparison with the PyTorch alpha). No change to
  product code.
- `docs/roadmap/alcedo_studio/edit/ai_mask_candidate/`: this report and the index.
- `docs/roadmap/alcedo_studio/edit/ai_segmentation_mask_master_plan.md`: E2 result in section
  8.16.4.
