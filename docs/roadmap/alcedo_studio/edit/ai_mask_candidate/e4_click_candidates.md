# E4: Click selection with candidates and hover highlight

**Issue:** #458, step E4. **Plan:** [AI Segmentation Mask master plan, section 8.16](../ai_segmentation_mask_master_plan.md#816-grid-holes-and-model-re-evaluation).
**Date:** 2026-10-19. **Branch:** `feature/ai-segmentation-sm1-quality-review`.
**Status:** measurements done. The user review of the sheets and the decision are open.

> **Images are local only.** The images under `images/` show photos that must not be shared
> again. They are in `.gitignore` and are not distributed with the GitHub repository. On a
> checkout without them, run the sheet command of the
> [script README](../../../../../scripts/ai_models/sam2/README.md#commands) with the local
> evidence under `build/tmp/explore/e4/`.

## 1. Question

E1 showed that SAM 2.1 removes the grid holes, but one click still selects a part: the car grille
instead of the car, the jacket instead of the person, the squirrel without the tail. One point does
not say whether the user wants the part or the whole object. Plan section 8.16.3 found that other
photo applications highlight the selection under the pointer before the click. Step E4 answers:

1. Does a selection from candidates remove the single-point ambiguity, and at what cost (time
   after the tool opens, GPU memory)?
2. Which interaction gives a direct response to the pointer and a stable highlight, with one model
   for the click, the point, and the box tools?

## 2. Result summary

- **Two designs were measured with the same model** (SAM 2.1 tiny and base plus, the E1 graphs):
  - **Precomputed candidates** (the plan text): when the tool opens, the point grid of the
    upstream `SAM2AutomaticMaskGenerator` gives a list of masks; the pointer looks up the masks
    under it.
  - **Live point decoder**: the point decoder runs at the pointer position (one positive point)
    and gives three masks of different size. The pointer highlights one of them; the user changes
    the size level.
- **Neither design selects the intended mask on every click by default, because the intent of a
  click is ambiguous.** On the 25 single-point prompts with a reference mask, 11 intend the whole
  object and 14 a part. A rule that prefers the larger mask helps the object prompts and breaks
  the part prompts. A level change (smaller or larger mask under the pointer) is necessary.
- **The live point decoder reaches the intended mask more often than the candidates of the same
  model.** Base plus live: default is the target on 20 of 25 prompts, one level step on 4, two
  steps on 1, not reachable on 0. Base plus candidates (32 x 32 grid): 18, 4, 1, 2. Tiny live:
  16, 4, 2, 3; tiny candidates: 17, 4, 0, 4. Together, the two sets do not reach more targets
  than the live decoder alone.
- **Cost of the candidates:** 0.83 s (16 x 16 grid) to 3.1 s (32 x 32) after the encoder, and
  1.3 GB more GPU memory with a batch of 16 prompts. With the batch of 64 prompts, the process
  needs 5.2 GB dedicated and 2.2 GB shared GPU memory, and the encoder becomes 17 times slower
  on the next photos (116 ms, then about 1.9 s), because the decoder memory moves the encoder
  buffers to shared memory. The upstream generator settings (predicted IoU > 0.8, stability
  >= 0.95, box non-maximum suppression 0.7) leave many click positions without a candidate
  (median coverage 0.44) and remove nested masks; the target is then a candidate on only 9 of
  25 prompts.
- **Cost of the live decoder:** the encoder runs once when the tool opens (DirectML warm p50:
  tiny 75 ms, base plus 159 ms). Each pointer position outside the current highlight runs the
  point decoder once (median 11.5 ms tiny, 11.9 ms base plus). No extra GPU memory over E1
  (tiny 1.36 GB, base plus 2.54 GB).
- **Hover stability.** If the decoder runs at every pointer position, a small pointer movement
  changes the highlight on some objects (`DSC_1456` car, tiny: same highlight on 25 percent of
  the nearby positions). Rule tested: **the highlight stays while the pointer is inside it; the
  decoder runs again only when the pointer leaves it.** With this rule the mean share of nearby
  positions with the same highlight rises from 0.88 to 0.96 (tiny) and from 0.91 to 0.98 (base
  plus); the candidates give 0.94 to 0.96. The decoder ran for 10 of 197 (tiny) and 5 of 198
  (base plus) nearby positions.
- **Recommended interaction (section 9):** live point decoder at the pointer, the highlight kept
  while the pointer is inside it, three size levels with the best predicted IoU as the default, a
  key or the mouse wheel to change the level, and a click that commits the highlighted mask.
  Precomputed candidates are not recommended: no gain in the reached targets, a wait of 0.8 to
  3.2 s, and more GPU memory.
- **Float16 (user decision, 2026-10-19):** accepted; a lower model precision for a shorter
  processing time. The float16 candidate graph takes 345 ms instead of 764 ms for 256 prompts and
  changes some candidate scores by up to 0.16 (section 6.2). The results of this report use
  float32; float16 forms of the encoder and the point decoder are not measured yet.
- **Recommended model:** base plus, because it reaches the target on all 25 prompts (tiny: 22).
  Tiny is the alternative if the GPU memory or the encoder time of base plus is too high. The
  user decides (decision D2).

## 3. Interaction designs

### 3.1 Precomputed candidates

1. When the tool opens: the encoder, then the decoder on a grid of N x N single positive points
   (`candidate_decoder.onnx`, batches of B prompts), the predicted IoU and stability filters, and
   duplicate removal. The result is a list of masks on the 256 x 256 model grid.
2. Pointer: the masks that contain the pointer position, from small to large, are the levels at
   that position. Masks with IoU >= 0.85 against the previous level are one level (the mask with
   the higher predicted IoU stays).
3. Default level: the largest, or the level with the best predicted IoU. A level change moves to
   the next smaller or larger level. A click commits the highlighted mask.

Two settings were measured:

| Setting | Predicted IoU | Stability | Duplicate removal |
| --- | --- | --- | --- |
| Upstream (`SAM2AutomaticMaskGenerator` defaults) | > 0.8 | >= 0.95 | box IoU > 0.7 |
| Nested (keeps the part and the whole) | > 0.5 | >= 0.85 | mask IoU > 0.85 |

### 3.2 Live point decoder

1. When the tool opens: the encoder (the E1 graph).
2. Pointer: the point decoder at the pointer position (`point_decoder.onnx`, one positive
   point). The levels are the three multimask masks (tokens 1 to 3), from small to large.
3. Default level: the best predicted IoU (the E1 selection) or the largest token. A level change
   moves to the next mask; the three masks are already computed, so the change has no model run.
4. Hover rule: the highlight stays while the pointer is inside the highlighted mask. The decoder
   runs again only for a position outside it.

## 4. Method

### 4.1 Inputs

| Item | Value |
| --- | --- |
| Photos and prompts | The SM1 quality set (`build/tmp/sm1/quality/`): 14 photos, the 26 single-point prompts of plan section 8.15. |
| Input size | The application gives the model at most 4096 px on the long edge (plan section 9.3, SM4; `kMaximumRasterMaskAxis`, P3). `make_input_set.py` scales every photo and every E1 reference mask to that limit (13 photos from 5.2 to 45.6 MP; result 2723 x 4096 to 4096 x 3074). All E4 runs read only this set. |
| Intended selection | `scripts/ai_models/sam2/click_targets.json`: per prompt, the intent (whole object or the part that the prompt names) and the reference. Object: the E1 PyTorch box mask of base plus. Part: the best E1 PyTorch multimask candidate of base plus at that point, which the E1 review found correct. `2088064441` person has no reference (the box holds the person, the board, and the bag); it is checked on the images only. |
| Models | SAM 2.1 tiny and base plus (E1 checkpoints and code, `facebookresearch/sam2` at `2b90b9f5`). |
| Graphs | `point_encoder.onnx` and `point_decoder.onnx`: byte-identical to E1 (same SHA-256). `candidate_decoder.onnx`: new, see section 5. |
| Scripts | [`scripts/ai_models/sam2/`](../../../../../scripts/ai_models/sam2/README.md) and `rust/puerh_mind/examples/segmentation_runtime_report.rs` (`candidates` mode) |
| Evidence | `build/tmp/explore/e4/` on the Windows machine (not committed): `input_4096/`, `onnx_*/`, `compare/`, `upstream/`, `runtime/`, `masks/`, `overlays/`, `probe/`, `profiles/` |

### 4.2 Machine

The machine of E1 and E2: Windows 11 build 22635, Intel Core i7-12700H, 31.8 GB, NVIDIA GeForce
RTX 3080 Laptop GPU 8 GB, driver 610.62. ONNX Runtime 1.24 through `ort` 2.0.0-rc.12 (Rust program,
production session settings of `ort_runtime::load_session`). PyTorch 2.14.1 (CPU), onnxruntime
1.31.0 (Python). The PyTorch checks ran with 4 threads. Only one measurement ran at a time.
macOS Core ML is not measured.

### 4.3 Metrics

- **Reference IoU:** the selected mask (256 x 256 logits, bilinear, threshold 0) against the
  reference mask, both at the photo scaled to a 1024 px long edge.
- **Default is the target:** the default level has reference IoU >= 0.85. **Level steps:** the
  distance in levels from the default to the level with the highest reference IoU, if that IoU is
  >= 0.85. **Not reachable:** no level has reference IoU >= 0.85.
- **Hover stability:** 8 positions on a circle around the prompt point (radius 1.5 percent of the
  short photo side, about 40 px at 4096 px), only the positions inside the reference mask. Share
  of these positions whose highlight has IoU >= 0.9 against the highlight at the centre.
- **Coverage:** share of the 256 x 256 model grid inside at least one candidate.
- **Time to candidates:** encoder + candidate decoder + host filters and duplicate removal.
  The scaling from 4096 px to the 1024 px model input (about 230 ms in the Rust program) is not
  included; the application renders the model input directly (SM4).
- GPU memory: peak `Dedicated Usage` and `Shared Usage` of the process
  (`run_with_gpu_memory.ps1`). The Rust program loads the encoder, the point decoder, and the
  candidate decoder together.

## 5. Candidate graph and agreement

`candidate_decoder.onnx` is the batched path of `SAM2AutomaticMaskGenerator._process_batch`:
`point_coords` [B, 1, 2] (one positive point per prompt) to `mask_logits` [B, 3, 256, 256] (tokens
1 to 3, clamped to +-32), `iou_scores` [B, 3], and `stability_scores` [B, 3]. `stability_scores` is
`calculate_stability_score` (threshold 0, offset 1.0) on the 256 x 256 logits; the upstream
generator uses the masks at the photo size. The Rust program applies the filters on the logits,
not on masks at the photo size.

`CandidateDecoderLogitsMatchPytorchWithinTolerance` (`compare_candidate_outputs.py`) compares the
graph on ONNX Runtime CPU with `SAM2ImagePredictor._predict` on two batches per photo. The
requirement applies to the masks that pass the generator filters in PyTorch (only these become
candidates): IoU >= 0.99 for each such mask and the same filter result for at least 99 percent of
all masks.

| Graph | Photos passed | Masks that pass the filters | Minimum IoU of these | Filter agreement, minimum | Maximum \|logit difference\| | Peak working set |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tiny, B = 16, float32 | 14 / 14 | 278 | 0.99997 | 1.000 | 0.0007 | 2,184 MB |
| base plus, B = 16, float32 | 14 / 14 | 340 | 1.00000 | 1.000 | 0.0032 | 2,561 MB |
| tiny, B = 64, float16 (option, section 6.2) | 13 / 14 | 1,055 | 0.993 | 0.984 (`DSC00728`) | 5.17 | 4,579 MB |

`CandidateDecoderOutputShapeMatchesExportedSpecification` passed for both 16-prompt graphs.
`SegmentationGraphsLoadWithProductionSessionSettings` passed on `directml` for all graphs.

**Comparison with the upstream generator** (`SAM2AutomaticMaskGenerator` with its defaults on the
photo scaled to 1024 px, tiny, PyTorch CPU; the Rust candidates with the upstream setting, 32 x 32,
on the 256 x 256 model grid): a mask counts as matched if the other set has a mask with IoU >= 0.8.

| Photo | Upstream masks | Rust masks | Upstream matched by Rust | Rust matched by upstream |
| --- | ---: | ---: | ---: | ---: |
| `DSC_1535` | 25 | 30 | 0.96 | 0.77 |
| `DSC00326` | 15 | 14 | 0.80 | 0.86 |
| `DSC00674` | 6 | 8 | 0.83 | 0.62 |
| `DSC00936` | 4 | 5 | 0.75 | 0.60 |
| `P1024439` | 10 | 19 | 0.80 | 0.58 |
| `DSC_1188` | 56 | 81 | 0.88 | 0.60 |
| `DSC00527` | 19 | 28 | 0.84 | 0.57 |
| `2088064441` | 10 | 15 | 0.80 | 0.60 |
| `DSC00728` | 23 | 36 | 0.87 | 0.50 |
| `5761824892` | 38 | 33 | 0.79 | 0.91 |
| `DSC00562` | 29 | 32 | 0.62 | 0.56 |
| `DSC00602` | 37 | 51 | 0.78 | 0.61 |
| `DSC_1456` | 53 | 64 | 0.77 | 0.62 |
| Nikon lupine | 9 | 30 | 0.89 | 0.27 |
| Median | | | 0.80 | 0.60 |

The Rust set has most of the upstream masks, plus more masks of its own. Two differences can
cause this; they are not separated: the Rust filters use the 256 x 256 logits (upstream: the masks
at the photo size), and the model inputs come from different resize implementations (E1, section
4). The
upstream setting is the first of the two settings in section 3.1. The results of section 7 use
the nested setting, which keeps more masks again.

## 6. Time and GPU memory (DirectML, RTX 3080 Laptop GPU)

### 6.1 Candidate decoder batch size

Decoder time for the 256 prompts of a 16 x 16 grid (tiny, float32), warm p50:

| Graph | B = 4 | B = 16 | B = 64 | B = 256 |
| --- | ---: | ---: | ---: | ---: |
| Dynamic prompt axis | 7,441 ms | 2,051 ms | 897 ms | 106,617 ms (more than 8 GB; DirectML pages through shared memory) |
| Fixed prompt count | — | 709 ms (median over the photos) | 764 ms | — |

- The dynamic graph places 146 nodes on the CPU provider. The fixed graph is one DirectML node.
- With the dynamic graph and B = 64, the profiled kernel time is 153 ms per run, but the run takes
  1,640 ms: the time is GPU execution, not host work.
- Per prompt, the decoder costs 2.8 ms (B = 16 fixed), against 10.5 ms for one point decoder run
  (E1). A larger batch does not make it much faster, and it needs much more memory (section 6.3).

### 6.2 Float16 (option, not used for the results)

A float16 form of the fixed 64-prompt graph (`--candidate-float16`; float32 inputs and outputs)
takes 345 ms for the 256 prompts (float32: 764 ms). It changes the predicted IoU and stability
scores by up to 0.16. On `DSC00728`, 3 of 192 masks then fall on the other side of a filter
threshold (agreement 0.984; required 0.99). The float32 fixed graph agrees exactly on the same
photo (maximum logit difference 0.0003). The results of this report use float32, because the
float16 decision was not made when they were measured.

**User decision (2026-10-19):** float16 is acceptable. A lower model precision for a shorter
processing time is accepted.

### 6.3 Time to candidates and GPU memory

| Model | Grid | Time to candidates, median / maximum | Candidate decoder, median | Host, median | Candidates, median (range) | Coverage, median / minimum |
| --- | ---: | --- | ---: | ---: | --- | --- |
| tiny | 16 x 16 | 828 / 907 ms | 709 ms | 30 ms | 60 (8-98) | 0.83 / 0.66 |
| tiny | 32 x 32 | 3,125 / 3,375 ms | 2,841 ms | 187 ms | 124 (16-231) | 0.91 / 0.77 |
| base plus | 16 x 16 | 913 / 1,156 ms | 700 ms | 48 ms | 58 (8-110) | 0.88 / 0.76 |
| base plus | 32 x 32 | 3,185 / 3,762 ms | 2,759 ms | 250 ms | 140 (17-257) | 0.94 / 0.89 |

Nested setting, B = 16. Upstream setting (tiny, B = 64): 11 (16 x 16) to 30 (32 x 32) candidates per
photo, median coverage 0.40 to 0.44, minimum 0.04.

| Process (graphs loaded) | Peak dedicated GPU memory | Peak shared GPU memory | Encoder on the next photos |
| --- | ---: | ---: | --- |
| tiny, encoder + point decoder (E1) | 1,355 MB | 237 MB | 76 ms |
| base plus, encoder + point decoder (E1) | 2,541 MB | 402 MB | 161 ms |
| tiny + candidate decoder, B = 16 | 2,626 MB | 375 MB | 82 ms (median) |
| base plus + candidate decoder, B = 16 | 3,815 MB | 540 MB | 169 ms (median) |
| tiny + candidate decoder, B = 64 | 5,235 MB | 2,151 MB | about 1,950 ms (first photo 116 ms) |

### 6.4 Pointer response

| Operation | tiny | base plus |
| --- | --- | --- |
| Encoder when the tool opens, warm p50 | 74.6 ms | 158.8 ms |
| Encoder session load (first use) | 1,102 ms | 1,610 ms |
| Point decoder at the pointer, median / maximum | 11.5 / 83.7 ms | 11.9 / 71.1 ms |
| Candidate lookup at the pointer, median / maximum (32 x 32) | 6.7 / 14.9 µs | 7.7 / 14.6 µs |
| Level change | no model run (both designs) | no model run |

## 7. Quality

### 7.1 Counts

25 single-point prompts with a reference (11 object, 14 part). Candidates: nested setting.

| Model | Selection | Default is the target | One level step | Two or more steps | Not reachable |
| --- | --- | ---: | ---: | ---: | ---: |
| tiny | Live, best predicted IoU | 16 | 4 | 2 | 3 |
| tiny | Live, largest | 12 | 6 | 4 | 3 |
| tiny | Candidates 16 x 16, best predicted IoU | 13 | 5 | 0 | 7 |
| tiny | Candidates 32 x 32, best predicted IoU | 17 | 4 | 0 | 4 |
| tiny | Candidates 32 x 32, largest | 16 | 3 | 2 | 4 |
| tiny | Candidates 32 x 32, upstream setting | 9 | 0 | 0 | 16 |
| base plus | **Live, best predicted IoU** | **20** | **4** | **1** | **0** |
| base plus | Live, largest | 13 | 9 | 3 | 0 |
| base plus | Candidates 16 x 16, best predicted IoU | 15 | 3 | 1 | 6 |
| base plus | Candidates 32 x 32, best predicted IoU | 18 | 4 | 1 | 2 |
| base plus | Candidates 32 x 32, largest | 14 | 8 | 1 | 2 |

### 7.2 Per prompt

Reference IoU of the default (best predicted IoU) and of the best level. Live: point decoder.
Candidates: nested setting, 32 x 32. Bold: not reachable (best level < 0.85).

| Photo | Prompt | Intent | tiny live default / best | tiny candidates default / best | base plus live default / best | base plus candidates default / best |
| --- | --- | --- | --- | --- | --- | --- |
| `DSC_1535` | torso | object | 0.53 / 0.95 | 0.53 / 0.95 | 0.54 / 0.99 | 0.54 / 0.99 |
| `DSC_1535` | hair | part | 0.96 / 0.96 | 0.96 / 0.96 | 0.99 / 0.99 | 0.98 / 0.98 |
| `DSC00326` | jacket | part | 0.96 / 0.96 | 0.96 / 0.96 | 1.00 / 1.00 | 1.00 / 1.00 |
| `DSC00326` | hair | part | 0.93 / 0.94 | 0.94 / 0.94 | 1.00 / 1.00 | 0.98 / 0.98 |
| `DSC00674` | face | part | 0.98 / 0.98 | 0.95 / 0.95 | 1.00 / 1.00 | 1.00 / 1.00 |
| `DSC00674` | hair | part | 0.91 / 0.91 | 0.91 / 0.91 | 1.00 / 1.00 | 0.67 / 0.99 |
| `DSC00936` | torso | object | 0.58 / 0.99 | 0.53 / 0.98 | 0.64 / 0.99 | 0.52 / 0.99 |
| `P1024439` | dress | object | 0.31 / 0.92 | **0.38 / 0.38** | 0.30 / 0.96 | **0.30 / 0.39** |
| `P1024439` | hair | part | 0.91 / 0.91 | 0.92 / 0.92 | 0.99 / 0.99 | 0.98 / 0.98 |
| `DSC_1188` | body | object | 0.58 / 0.97 | 0.96 / 0.96 | 0.99 / 0.99 | 0.99 / 0.99 |
| `DSC_1188` | tail | part | 0.96 / 0.96 | 0.94 / 0.94 | 1.00 / 1.00 | 0.97 / 0.97 |
| `DSC00527` | sky | part | 1.00 / 1.00 | 0.99 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |
| `DSC00527` | tree | part | 0.95 / 0.95 | 0.94 / 0.94 | 1.00 / 1.00 | 0.97 / 0.97 |
| `DSC00527` | kneeling woman | object | 0.48 / 0.93 | 0.48 / 0.91 | 0.50 / 0.97 | 0.49 / 0.97 |
| `2088064441` | board | part | **0.82 / 0.82** | **0.04 / 0.04** | 0.98 / 0.98 | **0.04 / 0.04** |
| `DSC00728` | balloon | object | 0.97 / 0.97 | 0.98 / 0.98 | 0.98 / 0.98 | 0.98 / 0.98 |
| `DSC00728` | sky | part | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `5761824892` | car | object | **0.01 / 0.78** | **0.01 / 0.01** | 0.01 / 0.94 | 0.96 / 0.96 |
| `5761824892` | kayak | part | 0.93 / 0.93 | 0.96 / 0.96 | 1.00 / 1.00 | 0.98 / 0.98 |
| `DSC00562` | duck | object | 0.97 / 0.97 | 0.97 / 0.97 | 0.99 / 0.99 | 0.99 / 0.99 |
| `DSC00602` | tram | object | 0.98 / 0.98 | 0.98 / 0.98 | 0.98 / 0.99 | 1.00 / 1.00 |
| `DSC_1456` | car | object | 0.03 / 0.98 | 0.98 / 0.98 | 0.99 / 0.99 | 0.98 / 0.98 |
| Nikon lupine | sky | part | 0.99 / 0.99 | 0.99 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |
| Nikon lupine | rock | object | 0.89 / 0.89 | 0.03 / 0.87 | 0.98 / 0.98 | 0.97 / 0.97 |
| Nikon lupine | flowers | part | **0.13 / 0.80** | **0.01 / 0.12** | 0.99 / 0.99 | **0.01 / 0.91** |

- **The E1 part selections are one level step away in the live design:** `DSC_1535` torso (jacket
  → person), `DSC00936` torso, `DSC_1188` body (tiny), `5761824892` car (base plus: grille → car),
  `DSC_1456` car (tiny), the kneeling woman, the bassist. The largest mask is the whole object in
  each of these cases.
- **Base plus reaches every target in the live design.** In all five base plus prompts where the
  default is not the target, the target is the largest of the three masks (the bassist needs two
  steps, the others one). Tiny does not reach the whole car (`5761824892`, best 0.78: the largest
  mask has a ragged lower edge), one lupine spike (0.80), and the whole paddle board (0.82).
- **Candidates miss masks that the point under the pointer gives:** the whole bassist
  (`P1024439`, 0.38), the board (`2088064441`: the candidate under the pointer is the bag,
  0.04), and the lupine spike. The grid points do not fall on these objects, or the filters remove
  their masks. In three cases the candidates make the whole object the default where the live
  default is a part (tiny `DSC_1188` body, tiny `DSC_1456` car, base plus `5761824892` car); the
  live design reaches these with one or two level steps.
- **No default rule fits both intents.** The best predicted IoU prefers parts; the largest mask
  prefers whole objects and selects the person instead of the hair (`DSC_1535` hair: 0.01 with
  the largest rule).

### 7.3 Sheets

All candidates of the nested setting (32 x 32), one colour per candidate, grey where no candidate
covers the photo:

![Candidate maps](images/e4_click/candidate_maps.jpg)

Highlight under the pointer per prompt and model: the three live point decoder levels, then the
candidate levels, each from small to large. Yellow frame: largest level; cyan frame: best predicted
IoU (the default). Under each tile: the area share and the reference IoU. The images come from a
second DirectML run with the same settings (32 x 32 grid only), which also writes the three live
levels; its selections are the same as in the measurement run.

**`DSC_1535`** (torso: jacket by default, person one level up):

![DSC_1535](images/e4_click/DSC_1535.jpg)

**`5761824892`** (car: grille by default; base plus gives the whole car one level up, tiny the car
with a ragged lower edge; the kayak point gives the car at the largest level):

![5761824892](images/e4_click/5761824892.jpg)

**`DSC_1188`** (squirrel body and tail):

![DSC_1188](images/e4_click/DSC_1188.jpg)

**`P1024439`** (bassist: two levels up in the live design; no candidate holds the whole bassist):

![P1024439](images/e4_click/P1024439.jpg)

**`2088064441`** (paddle boarder: the person has no reference; the candidate under the board point
is the bag):

![2088064441](images/e4_click/2088064441.jpg)

The other photos: [`DSC00326`](images/e4_click/DSC00326.jpg),
[`DSC00674`](images/e4_click/DSC00674.jpg), [`DSC00936`](images/e4_click/DSC00936.jpg),
[`DSC00527`](images/e4_click/DSC00527.jpg), [`DSC00728`](images/e4_click/DSC00728.jpg),
[`DSC00562`](images/e4_click/DSC00562.jpg), [`DSC00602`](images/e4_click/DSC00602.jpg),
[`DSC_1456`](images/e4_click/DSC_1456.jpg),
[Nikon lupine](images/e4_click/Nikon-D800e-raw-00007.jpg).

## 8. Hover stability

Mean share of the nearby pointer positions with the same highlight (25 prompts; the median is
1.00 for every selection):

| Model | Live, decoder at every position | Live, highlight kept inside | Candidates 16 x 16 | Candidates 32 x 32 |
| --- | ---: | ---: | ---: | ---: |
| tiny | 0.88 | 0.96 | 1.00 | 0.94 |
| base plus | 0.91 | 0.98 | 0.99 | 0.96 |

Prompts with a changing live highlight: `DSC_1456` car (tiny 0.25 → 0.50 with the rule),
`5761824892` car (base plus 0.25 → 0.50), `2088064441` person (0.00 to 0.17 → 1.00), `DSC_1188`
tail and body (0.50 to 0.62 → 1.00), the balloon and the bassist (0.75 to 0.88 → 1.00); the
kneeling woman stays at 0.88. On the two cars, half of the nearby positions are outside the
highlight at the centre, and the decoder gives a different mask there.

## 9. Recommendation

| Item | Recommendation | Evidence |
| --- | --- | --- |
| Selection design | Live point decoder at the pointer | Reaches more targets than the candidates of the same model (section 7.1), no wait after the encoder, no extra GPU memory |
| Hover rule | Keep the highlight while the pointer is inside it; run the decoder for a position outside it, with at most one run at a time and the latest pointer position next | Same-highlight share 0.96 to 0.98; 2.5 to 5 percent of the nearby positions need a decoder run |
| Levels | The three masks under the pointer, from small to large; default: best predicted IoU; mouse wheel or a key to change the level | 20 of 25 targets by default (base plus), the rest within two steps |
| Click | Commit the highlighted mask (the logits that the hover computed; no new run). Add and subtract points and the box tool use the same point decoder (E1). | — |
| Precomputed candidates | Not recommended | No gain in reached targets; 0.8 to 3.2 s after the encoder; +1.3 GB GPU memory; 1 to 34 percent of the photo without a candidate on the 16 x 16 grid |
| Model | base plus; tiny as the lower-memory alternative | base plus: 25 of 25 reachable, encoder 159 ms, 2.54 GB. tiny: 22 of 25, 75 ms, 1.36 GB |

The user decides the model and the interaction set (decision D2, step E6).

## 10. Open items and next steps

- **User review** of the sheets in section 7.3.
- **Intent of the default level.** The default is a part on 5 of 25 prompts (base plus) where the
  user wants the whole object. A remembered level (after a level change, the next highlight uses
  the same relative level) is not measured.
- **Tiny failures:** the whole car with the kayak, one lupine spike, and the paddle board are not
  reachable with tiny. Several points (E1) reach them.
- **UI response** in the application (QSG overlay, 256 x 256 mask scaled to the view, pointer
  event rate) is not measured. The decoder time (12 ms) is below one frame at 60 Hz only on this
  GPU.
- **macOS Core ML** is not measured.
- **Float16 of the live design:** the user accepts float16 (section 6.2). The float16 forms of
  `point_encoder.onnx` and `point_decoder.onnx` (time, GPU memory, the counts of section 7.1) are
  not measured.
- **Edges:** the masks are the 256 x 256 logits upsampled; hair and fur need the edge stage (E5).

## 11. Changed files

- `scripts/ai_models/sam2/`: `sam2_graphs.py` (`CandidateDecoder`, `grid_points`),
  `export_onnx.py` (`candidate_decoder.onnx`, `--graphs`, `--candidate-batch`,
  `--candidate-float16`), new `compare_candidate_outputs.py`, `make_input_set.py`,
  `click_targets.json`, `probe_candidates.py`, `make_candidate_sheets.py`, `README.md`.
- `rust/puerh_mind/examples/segmentation_runtime_report.rs`: new `candidates` mode (candidate
  generation, levels under the pointer, live point decoder, hover stability, overlays). No change
  to product code.
- `docs/roadmap/alcedo_studio/edit/ai_mask_candidate/`: this report and the index.
- `docs/roadmap/alcedo_studio/edit/ai_segmentation_mask_master_plan.md`: E4 result in section
  8.16.4.
