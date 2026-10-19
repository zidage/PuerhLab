# E1: SAM 2.1 point and box prompts

**Issue:** #458, step E1. **Plan:** [AI Segmentation Mask master plan, section 8.16](../ai_segmentation_mask_master_plan.md#816-grid-holes-and-model-re-evaluation).
**Date:** 2026-10-19. **Branch:** `feature/ai-segmentation-sm1-quality-review`.
**Status:** measurements done. The user review of the visual results and the decision are open.

> **Images are local only.** The images under `images/` show photos that must not be shared
> again. They are in `.gitignore` and are not distributed with the GitHub repository. On a
> checkout without them, run the sheet command of the
> [script README](../../../../../scripts/ai_models/sam2/README.md#commands) with the local
> evidence under `build/tmp/explore/e1/`.

## 1. Question

Plan section 8.16.1 found that the EfficientSAM3 point and box masks have regular grid holes and
fail on the close-up portrait. The cause is the student image encoder (native 32 x 32 output
resized to 72 x 72, decoder logits near 0). Step E1 answers two questions:

1. Does an encoder with native multi-scale features (SAM 2.1 Hiera, stride 4, 8, and 16 features)
   remove the grid holes and the close-up failure?
2. Which size (tiny, small, base plus, large) is affordable on the DirectML machine?

## 2. Result summary

- **Grid holes: removed.** On all 48 point and box prompts of the quality set, the SAM 2.1 masks
  have no regular hole pattern. The largest grid energy is 0.04 (tiny) to 0.11 (small), compared
  with 0.54 for EfficientSAM3 EfficientViT-M. 12 of 48 EfficientSAM3 prompts have a grid energy
  above 0.1; 0 to 1 of 48 SAM 2.1 prompts do.
- **Close-up portrait: corrected.** The `DSC00674` box gives the whole person in all four sizes
  (EfficientSAM3: face left out, holes). The share of uncertain logits drops from 0.83 to 0.06 to
  0.26.
- **Other EfficientSAM3 failures: corrected.** The sky points (`DSC00728`, `DSC00527`, the Nikon
  lupine photo), the legs in the `DSC00326` box, and the Nikon rock and flower points are correct
  in all sizes.
- **Single-point ambiguity: not removed.** One point still selects a part: `DSC_1535` torso gives
  the jacket only, the `5761824892` car point gives the grille, the `DSC_1188` body point misses
  the tail in tiny and small. This is the subject of step E4 (candidates computed before the
  click), not of the encoder.
- **Cost:** the encoder is slower than EfficientSAM3 (DirectML warm p50: tiny 76 ms, small 87 ms,
  base plus 161 ms, large 322 ms; EfficientSAM3 46 ms). The decoder is faster (10.4 to 10.7 ms;
  EfficientSAM3 12.9 ms). Peak dedicated GPU memory: tiny 1.36 GB, small 1.38 GB, base plus
  2.54 GB, large 3.10 GB (EfficientSAM3 1.00 GB).
- **License:** Apache-2.0 for the code and the checkpoints. The Meta SAM License of the SAM3
  weights in EfficientSAM3 is not necessary.
- **Recommendation:** SAM 2.1 tiny as the point and box model, and as the candidate generator of
  step E4. Base plus is the quality alternative if E4 shows that tiny is not sufficient. Small
  gives no quality gain over tiny on this set. Large is not affordable (852 MB encoder file, 3.1 GB
  GPU memory, 322 ms) and gives no visible gain over base plus. The user decides (decision D2).

## 3. Method

### 3.1 Inputs

| Item | Value |
| --- | --- |
| Photos and prompts | The SM1 quality set (`build/tmp/sm1/quality/`): 14 photos, 48 point and box prompts (plan section 8.15). Text prompts do not apply to SAM 2.1. |
| Baseline | EfficientSAM3 EfficientViT-M (`efficient_sam3_efficientvit_m_geo_ft.pt`), the SM1 ONNX graphs, measured again in the same session. |
| Code | `facebookresearch/sam2` at `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| Checkpoints | `sam2.1_hiera_{tiny,small,base_plus,large}.pt` (release 2024-09-28) |
| Scripts | [`scripts/ai_models/sam2/`](../../../../../scripts/ai_models/sam2/README.md) and `rust/puerh_mind/examples/segmentation_runtime_report.rs` |
| Evidence | `build/tmp/explore/e1/` on the Windows machine (not committed): `onnx/`, `compare/`, `probe/`, `runtime/`, `profiles/`, `overlays/`, `sheets/` |

### 3.2 Machine

Windows 11 build 22635, Intel Core i7-12700H, 31.8 GB, NVIDIA GeForce RTX 3080 Laptop GPU 8 GB,
driver 610.62. ONNX Runtime 1.24 through `ort` 2.0.0-rc.12 (Rust program, production session
settings of `ort_runtime::load_session`). PyTorch 2.14.1 (CPU), onnxruntime 1.31.0 (Python).
macOS Core ML is not measured; plan section 8.16.4 measures it after the user selects a candidate.

### 3.3 Export

The SAM 2.1 graphs have the same inputs and outputs as the SM1 EfficientSAM3 point and box graphs
(see the script README), so the Rust program runs both without a separate code path. The Rust
program now reads the input size, the mask size, and the normalization from `model_input` in
`export_report.json` (SAM 2.1: 1024 px, 256 px masks, ImageNet mean and std; EfficientSAM3:
1008 px, 288 px masks, mean and std 0.5). The input is the whole photo stretched to 1024 x 1024,
the preprocessing of `SAM2ImagePredictor`.

### 3.4 Metrics

- `OnnxMaskLogitsMatchPytorchWithinTolerance`: every mask of `SAM2ImagePredictor.predict`
  (3 multimask and 1 single mask per prompt) compared with the ONNX Runtime CPU mask; IoU at the
  original photo size, required 0.99.
- Probe metrics (`probe_masks.py`, plan section 8.16.1) on the mask that the overlay shows (the
  best multimask token for one point, else the single-mask selection):
  - **uncertain:** share of low-resolution logits with |logit| < 2;
  - **grid energy:** share of the high-pass logit spectrum on the lines of the embedding cell
    period (4 logit pixels: frequencies 64 and 128 for SAM 2.1, 72 and 144 for EfficientSAM3).
    With the plan line set (72, 144, 32, 64, 96) the script reproduces the plan values for
    EfficientSAM3 exactly (0.826 / 0.288, 0.213 / 0.374, 0.012 / 0.031, 0.049 / 0.022);
  - **hole and island fraction:** enclosed holes and foreground outside the largest component,
    relative to the mask area, at a 900 px long edge. A legitimate gap (arms, a stand in front of
    the subject, three separate ducks) also counts, so these two values need the images.
- Latency: `segmentation_runtime_report runtime` on DirectML, load, first inference, warm encoder
  (30 runs) and decoder (100 runs), p50 / p95 / maximum. CPU: one photo, 10 encoder and 30 decoder
  runs.
- GPU memory: peak `Dedicated Usage` and `Shared Usage` of the process, Windows `GPU Process Memory`
  counters, sampled every 250 ms (`run_with_gpu_memory.ps1`), from a run without the profiling
  pass. Process memory: the private bytes and the peak working set that the Rust program reports.

## 4. Export and agreement

| Variant | Parameters (image encoder) | Checkpoint bytes | `point_encoder.onnx` bytes | `point_decoder.onnx` bytes | Agreement (passed / compared) | Minimum IoU | Maximum \|logit difference\| |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| tiny | 39.0M (27.2M) | 156,008,466 | 109,381,695 | 21,188,206 | 192 / 192 | 0.99999 | 0.0003 |
| small | 46.1M (34.3M) | 184,416,285 | 137,928,565 | 21,188,206 | 192 / 192 | 0.99998 | 0.0005 |
| base plus | 80.9M (69.1M) | 323,606,802 | 277,382,455 | 21,188,206 | 192 / 192 | 0.99997 | 0.0030 |
| large | 224.4M (212.7M) | 898,083,611 | 852,360,706 | 21,188,206 | 192 / 192 | 0.99974 | 0.0595 |

All graphs export with opset 17 and the TorchScript exporter without an error.
`DecoderOutputShapeMatchesExportedSpecification` passed for all four variants
(`mask_logits` [1, 4, 256, 256], `iou_scores` [1, 4], `object_score_logits` [1, 1]).
`SegmentationGraphsLoadWithProductionSessionSettings` passed on `directml` and `cpu`.

SHA-256 of the ONNX files:

| File | SHA-256 |
| --- | --- |
| tiny `point_encoder.onnx` | `17a554b420d3c51a109567f784ff3792b354c78dc8c2c09c585403329b412d61` |
| tiny `point_decoder.onnx` | `6f9a0ef2c217a0abccc8b0c01e631c42c6a78a2038a7944cc7fcb4f536439278` |
| small `point_encoder.onnx` | `6f202bce40bf4829846724a3cccbf76be40b49a0f51700b69c6b820b34d07fcb` |
| small `point_decoder.onnx` | `d34092e234387a232ba06c26d8b846443ac5922be71bf88f1096358f7456c90c` |
| base plus `point_encoder.onnx` | `81ebb9e206fd5dab68a24084b408663dc8faaba2118f4536d69281dc35b4e3ea` |
| base plus `point_decoder.onnx` | `5d246a28971eaf2874f1117335eea31663b970126e45884c60afdb4da6ce1277` |
| large `point_encoder.onnx` | `d48fe2668e716ea96c613999d2daeee3ea65af34b61aa6f6f7d9abc493b5f0b1` |
| large `point_decoder.onnx` | `e0f338aa69e5d1132a7251553e548942c2dd9082cf63b295a3f80515b09e1b6d` |
| SAM 2 `LICENSE` (Apache-2.0) | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

**Rust pipeline compared with PyTorch.** The Rust program resizes with `image::imageops`
(Triangle); PyTorch uses torchvision bilinear with antialias. For the mask that the overlay
shows, 46 of 48 (tiny, small) and 48 of 48 (base plus, large) prompts have IoU of at least 0.99;
the minimum is 0.958 (small, `DSC00936` torso). Over all 192 masks the median IoU is 0.998, and
some multimask candidates that are not shown change more (minimum 0.40, large, `DSC00936` torso,
token 1). Result: a candidate that the model ranks low can change with a small resize difference.
The product must use one resize implementation for the encoder input on all platforms.

## 5. Quality

### 5.1 Probe metrics

Over the 48 point and box prompts:

| Model | Uncertain, median | Uncertain, mean | Prompts with uncertain > 0.1 | Grid energy, median | Grid energy, 90th percentile | Grid energy, maximum | Prompts with grid energy > 0.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientSAM3 EfficientViT-M | 0.0135 | 0.116 | 12 | 0.028 | 0.312 | 0.538 | 12 |
| SAM 2.1 tiny | 0.0029 | 0.022 | 3 | 0.011 | 0.026 | 0.041 | 0 |
| SAM 2.1 small | 0.0032 | 0.024 | 3 | 0.010 | 0.031 | 0.114 | 1 |
| SAM 2.1 base plus | 0.0023 | 0.013 | 2 | 0.010 | 0.023 | 0.074 | 0 |
| SAM 2.1 large | 0.0026 | 0.014 | 2 | 0.010 | 0.021 | 0.060 | 0 |

The prompts of plan section 8.16.1 (uncertain / grid energy):

| Prompt | EfficientSAM3 | tiny | small | base plus | large |
| --- | --- | --- | --- | --- | --- |
| `DSC00674` box (close-up) | 0.826 / 0.280 | 0.180 / 0.027 | 0.257 / 0.051 | 0.057 / 0.012 | 0.172 / 0.019 |
| `DSC00728` sky point | 0.213 / 0.360 | 0.007 / 0.023 | 0.007 / 0.018 | 0.005 / 0.013 | 0.005 / 0.017 |
| `DSC00326` box | 0.012 / 0.019 | 0.002 / 0.006 | 0.001 / 0.008 | 0.001 / 0.008 | 0.001 / 0.008 |
| `DSC_1535` box | 0.049 / 0.012 | 0.013 / 0.007 | 0.011 / 0.006 | 0.010 / 0.006 | 0.007 / 0.007 |
| Nikon sky point | 0.002 / 0.478 (empty mask) | 0.008 / 0.033 | 0.021 / 0.033 | 0.006 / 0.028 | 0.007 / 0.036 |
| `DSC00602` tram point | 0.238 / 0.224 | 0.152 / 0.039 | 0.146 / 0.114 | 0.106 / 0.074 | 0.071 / 0.060 |

The `DSC00674` box and the three-point prompt on the same photo stay the most uncertain prompts for
SAM 2.1 (0.16 to 0.34), but the uncertain pixels are inside the subject (the chest and the jacket),
and the thresholded mask is the whole person.

### 5.2 Logits

Selected low-resolution logits (blue negative, white near 0, red positive; square model space).
EfficientSAM3 has wide white areas and a regular cell pattern. SAM 2.1 logits are decisive and
follow the image edges.

![Selected logits of six key prompts](images/e1_sam2_1/logits_key_prompts.jpg)

Residual artifacts in SAM 2.1: large shows isolated single-cell dots (stride 16) inside confident
regions (face, sky); they do not cross 0, so the mask does not change. Small has a local dot
pattern on the hair in the `DSC00674` three-point prompt (see 5.3); this is the only visible hole
pattern of SAM 2.1 on the set.

### 5.3 Masks per photo

Each sheet has one row per prompt and one column per model. Overlays are from the Rust program on
DirectML with production session settings: blue is the mask, green dots are positive points, red
dots are negative points, the yellow rectangle is the box. Single points show the best multimask
candidate; boxes and several points show the single-mask selection.

**Case 7, `DSC00674` (close-up portrait, ISO 1250).** The main EfficientSAM3 failure. SAM 2.1
box: whole person in all sizes. Face point: face only (small) or face with the neck and chest skin
(tiny, base plus, large). Hair point: hair; tiny and small leave a few holes at the left bun.
Three points: small has the dot pattern on the hair.

![DSC00674](images/e1_sam2_1/DSC00674.jpg)

**Case 1, `DSC_1535` (portrait, busy street).** Box, three points, hair: correct in all models.
Torso point: jacket only in all SAM 2.1 sizes (single-point ambiguity).

![DSC_1535](images/e1_sam2_1/DSC_1535.jpg)

**Case 2, `DSC00326` (portrait from behind, waterfall).** SAM 2.1 box and three points include
the legs (EfficientSAM3: holes in the legs).

![DSC00326](images/e1_sam2_1/DSC00326.jpg)

**Case 10, `DSC00936` (person against bright mist).** Box and three points correct in all
models. Torso point: tiny, small, and base plus give the body without the shorts; large gives the
whole person.

![DSC00936](images/e1_sam2_1/DSC00936.jpg)

**Extra, `P1024439` (backlit hair, microphone stand in front).** SAM 2.1 gives the person
without the bass guitar; the stand splits the mask into parts (island fraction 0.36 to 0.45,
legitimate). EfficientSAM3 includes the bass. The negative point on the stand has no visible
effect in any model.

![P1024439](images/e1_sam2_1/P1024439.jpg)

**Case 3, `DSC_1188` (squirrel, fur).** Box and body-and-tail points: whole squirrel in all
models. Tail point: tail only in SAM 2.1 (EfficientSAM3: whole squirrel with a grid on the
rock). Body point: tiny and small miss the tail; base plus and large include it.

![DSC_1188](images/e1_sam2_1/DSC_1188.jpg)

**Case 4, `DSC00527` (sky, tree horizon).** Sky point: whole sky with a clean tree edge in all
SAM 2.1 sizes. Tree point and the kneeling woman: correct in all models.

![DSC00527](images/e1_sam2_1/DSC00527.jpg)

**Case 6, `DSC00728` (balloon, 0.4 percent of the frame).** Balloon point and box: correct.
Sky point: whole sky without the EfficientSAM3 gap above the mountains and without the false
patch on the cliff.

![DSC00728](images/e1_sam2_1/DSC00728.jpg)

**Case 5, `2088064441` (paddle boarder, 1.3 percent of the frame).** Box: person, board, and bag
with a clean outline (EfficientSAM3: one blob with the water). Board point: whole board (tiny,
base plus) or the bag on the board only (small, large). Person point: body parts (the shirt in
large). See the detail crop in section 5.5.

![2088064441](images/e1_sam2_1/2088064441.jpg)

**Case 8, `5761824892` (45.6 MP, car with kayak).** Box: whole car in all models. Kayak point:
correct in all models. Car point: the grille only in all models (single-point ambiguity).

![5761824892](images/e1_sam2_1/5761824892.jpg)

**Extra, `Nikon-D800e-raw-00007` (lupine field, mountain horizon).** Sky point: whole sky in all
SAM 2.1 sizes (EfficientSAM3: empty mask). Rock point and rock box: the rock (EfficientSAM3 point:
the whole field with grid artifacts). Flower point: one lupine spike (tiny and small smaller than
base plus and large; EfficientSAM3: the whole field).

![Nikon-D800e-raw-00007](images/e1_sam2_1/Nikon-D800e-raw-00007.jpg)

**Case 11, `DSC00562` (ducks).** One duck and the duck box: correct. Three duck points: the
ducks on the wall, without the hedge (EfficientSAM3 includes the hedge). See the detail crop in
section 5.5.

![DSC00562](images/e1_sam2_1/DSC00562.jpg)

**Case 12, `DSC00602` (tram, 42.3 MP).** Box: whole tram; the bottom edge near the shadow is
ragged in all models. Point: small and base plus add small islands on the canopy at the top
right.

![DSC00602](images/e1_sam2_1/DSC00602.jpg)

**Extra, `DSC_1456` (car, pedestrians).** Box: whole car. Point: whole car, except tiny, which
selects the grille.

![DSC_1456](images/e1_sam2_1/DSC_1456.jpg)

### 5.4 Result per case

| Case | Photo | EfficientSAM3 EfficientViT-M | SAM 2.1 (all sizes unless stated) |
| --- | --- | --- | --- |
| 1 | `DSC_1535` | Box, 3 points correct; torso point: suit with a gap | Box, 3 points, hair correct; torso point: jacket only |
| 2 | `DSC00326` | Box: holes in the legs | Box, 3 points with legs; hair and jacket correct |
| 3 | `DSC_1188` | Tail point: whole squirrel; body point: no tail | Tail point: tail; body point: no tail (tiny, small), with tail (base plus, large) |
| 4 | `DSC00527` | Sky coarse at the horizon | Sky with a clean tree edge |
| 5 | `2088064441` | Box: person and board as one blob with the water | Box: person, board, and bag, clean outline; board point: whole board (tiny, base plus) or the bag only (small, large); person point: body parts |
| 6 | `DSC00728` | Sky point: gap and a false patch with a grid | Sky correct; balloon correct |
| 7 | `DSC00674` | Box, points: face left out, grid holes | Box: whole person; face, hair correct (small: dots on the hair with 3 points) |
| 8 | `5761824892` (45.6 MP) | Box: whole car; kayak correct | Box: whole car; kayak correct; car point: grille only |
| 10 | `DSC00936` | Box correct, edge about 20 px outside | Box correct with a closer edge; torso point without the shorts (except large) |
| 11 | `DSC00562` | 3 points: ducks and the hedge | 3 points: ducks without the hedge |
| 12 | `DSC00602` (42.3 MP) | Box: ragged bottom edge | Box: ragged bottom edge; point: islands on the canopy (small, base plus) |
| extra | `DSC_1456` | Box and point correct | Box correct; point correct except tiny (grille) |
| extra | `P1024439` | Box, 3 points: bassist with the bass | Box, 3 points: bassist without the bass, split by the stand |
| extra | Nikon lupine | Sky point: empty mask; rock and flower points: whole field | Sky, rock, flowers correct |

### 5.5 Edges

Crops of the 2048 px overlays (bilinear upsampling of the 256 x 256 logits, threshold 0; no edge
refinement):

![Detail crops](images/e1_sam2_1/edge_detail.jpg)

- The SAM 2.1 edge follows the subject more closely than EfficientSAM3 (`DSC00936`, `DSC00326`
  legs, the squirrel outline).
- Single hairs and fur tips are not in any mask. The mask resolution (256 x 256 for the whole
  photo, one logit pixel is 16 to 24 photo pixels on a 24 MP photo) does not allow it. An edge
  stage (step E2 matting models or step E5) is still necessary.

## 6. Performance

### 6.1 DirectML (RTX 3080 Laptop GPU)

Milliseconds. Warm values are p50 / p95 / maximum. The second warm encoder value is a repeat run
without the profiling pass.

| Model | Encoder load | Encoder first run | Encoder warm | Encoder warm, repeat (p50 / p95) | Decoder load | Decoder warm | Peak dedicated GPU memory | Peak shared GPU memory | Private bytes | Peak working set |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| EfficientSAM3 EfficientViT-M | 700 | 75 | 45.9 / 49.9 / 55.4 | — | 208 | 12.9 / 14.5 / 22.1 | 995 MB | 215 MB | 1,429 MB | 860 MB |
| SAM 2.1 tiny | 1,471 | 116 | 75.9 / 81.6 / 82.2 | 76.3 / 81.7 | 185 | 10.4 / 11.3 / 11.8 | 1,355 MB | 237 MB | 1,807 MB | 789 MB |
| SAM 2.1 small | 1,161 | 130 | 87.3 / 93.0 / 93.1 | 87.3 / 93.3 | 198 | 10.3 / 11.3 / 12.3 | 1,382 MB | 265 MB | 1,866 MB | 815 MB |
| SAM 2.1 base plus | 1,471 | 237 | 160.5 / 161.8 / 172.7 | 160.7 / 165.4 | 190 | 10.6 / 11.5 / 13.3 | 2,541 MB | 402 MB | 3,171 MB | 954 MB |
| SAM 2.1 large | 2,487 | 465 | 321.8 / 322.9 / 323.1 | 322.2 / 323.4 | 190 | 10.7 / 12.0 / 12.7 | 3,101 MB | 961 MB | 4,300 MB | 1,524 MB |

The EfficientSAM3 row is a new run with only the point and box graphs; the SM1 record
(47.7 / 92.5 ms) also had the text graphs loaded. Its p50 is the same.

**Operator placement (DirectML, production settings).** The SAM 2.1 encoder runs completely on
DirectML (one fused DirectML node, 0 CPU nodes) in all sizes. The decoder places 121 nodes on the
CPU provider (Gather 37, Unsqueeze 22, Sqrt 14, Concat 11, Slice 10, Cast 7, Div 7, Mul 7,
Reshape 3, Equal 1, OneHot 1, Where 1), 3.2 to 3.4 ms of kernel time in two profiled runs. This is
the same list as the EfficientSAM3 decoder, because SAM3 reuses the SAM 2 decoder.

### 6.2 CPU (i7-12700H)

One photo (`DSC_1535`), 10 encoder and 30 decoder runs, production session settings.

| Model | Encoder warm p50 / p95 | Decoder warm p50 / p95 | Peak working set |
| --- | --- | --- | ---: |
| EfficientSAM3 EfficientViT-M | 758 / 837 | 48.6 / 69.2 | 576 MB |
| SAM 2.1 tiny | 1,118 / 1,156 | 34.8 / 40.7 | 835 MB |
| SAM 2.1 small | 1,283 / 1,317 | 33.8 / 40.9 | 865 MB |
| SAM 2.1 base plus | 2,277 / 2,354 | 32.4 / 40.4 | 1,319 MB |
| SAM 2.1 large | 5,135 / 5,781 | 33.7 / 43.0 | 1,954 MB |

## 7. License

| Item | License | Commercial use |
| --- | --- | --- |
| SAM 2 code (`facebookresearch/sam2`) | Apache-2.0 | yes |
| SAM 2.1 checkpoints | Apache-2.0 (README of the repository: "The SAM 2 model checkpoints ... are licensed under Apache 2.0") | yes |
| Optional `cc_torch` connected components (not used by the graphs) | `LICENSE_cctorch` (BSD-3-Clause) | not shipped |

A published model directory needs the Apache-2.0 text. The Meta SAM License is not necessary for
SAM 2.1, which removes the license item of #454.

## 8. Recommendation per size

| Size | Quality on the set | DirectML encoder p50 | Files (encoder + decoder) | GPU memory | Recommendation |
| --- | --- | ---: | ---: | ---: | --- |
| tiny | No grid holes; close-up correct; two single points select a part (car grille, squirrel body) | 76 ms | 131 MB | 1.36 GB | **Recommended** default for point, box, and the E4 candidates |
| small | Same as tiny; one local dot pattern (`DSC00674` hair, 3 points); canopy islands | 87 ms | 159 MB | 1.38 GB | Not recommended: no gain over tiny |
| base plus | Most consistent single points (squirrel with tail, whole car) | 161 ms | 299 MB | 2.54 GB | Quality alternative if E4 shows that tiny candidates are not sufficient |
| large | Same as base plus; isolated logit dots; torso point gives the whole person | 322 ms | 874 MB | 3.10 GB | Not recommended: file size, memory, and latency |

## 9. Open items and next steps

- **User review.** The user reviews the sheets in section 5 and records the result per case.
- **Single-point ambiguity** remains; step E4 measures the SAM 2.1 automatic mask generator with
  tiny (and base plus for comparison): candidate time for a point grid, and whether candidate
  selection removes the part selections listed in section 2.
- **Edges.** Single hairs and fur are not in the masks; steps E2 and E5.
- **Aspect ratio.** The input is stretched to a square like EfficientSAM3. A panorama (case 9) is
  not in the set. The letterbox result of SM1 (`compare_preprocessing.py`) is not repeated for
  SAM 2.1.
- **Resize implementation.** The product resize must give the same encoder input on Windows and
  macOS (section 4).
- **macOS Core ML** is measured after the user selects a size (plan section 8.16.4).

## 10. Changed files

- `scripts/ai_models/sam2/`: `sam2_graphs.py`, `export_onnx.py`, `compare_onnx_outputs.py`,
  `probe_masks.py`, `run_with_gpu_memory.ps1`, `make_comparison_sheets.py`, `requirements.txt`,
  `README.md`.
- `rust/puerh_mind/examples/segmentation_runtime_report.rs`: the input size, the mask size, and the
  normalization come from `model_input` in `export_report.json` (default: the EfficientSAM3
  values). No change to product code.
- `docs/roadmap/alcedo_studio/edit/ai_mask_candidate/`: this report, the index, and a
  `.gitignore` for `images/` (the images stay local; see the note at the top).
