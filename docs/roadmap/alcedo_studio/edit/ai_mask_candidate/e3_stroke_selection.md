# E3: Stroke selection

**Issue:** #458, step E3. **Plan:** [AI Segmentation Mask master plan, section 8.16](../ai_segmentation_mask_master_plan.md#816-grid-holes-and-model-re-evaluation).
**Date:** 2026-10-20. **Branch:** `feature/ai-segmentation-sm1-quality-review`.
**Status:** measurements done. The user review of the sheets and the decision are open.

> **Images are local only.** The images under `images/` show photos that must not be shared
> again. They are in `.gitignore` and are not distributed with the GitHub repository. On a
> checkout without them, run the sheet command of the
> [script README](../../../../../scripts/ai_models/sam2/README.md#e3-stroke-selection) with the
> local evidence under `build/tmp/explore/e3/`.

## 1. Question

E4 showed that one click is ambiguous: the click on the torso selects the jacket, and the user
needs a level change to get the person. The user assessed on 2026-10-20 that a selection that
follows the pointer (the E4 hover highlight) is probably not a good interaction. Step E3 measures
stroke selection, like the Quick Selection tool of Photoshop: the user drags a stroke inside the
region, and points sampled along the stroke are positive points of the SAM 2.1 point decoder; a
stroke with a modifier key gives negative points. Step E3 answers:

1. Does a short stroke inside the region select the intended mask without a level change?
2. How many points along the stroke, at which spacing?
3. Do add and subtract strokes correct a wrong selection, and how are the strokes combined?
4. What does one stroke cost in decoder time?

## 2. Result summary

- **A short stroke does not remove the ambiguity.** A stroke selects the region that it covers. A
  stroke over half of the region (half of the inner chord through the click point, 5 points)
  reaches the intended mask on 18 of 25 prompts with base plus (one click: 20) and 17 with tiny
  (one click: 16). On the torso of `DSC_1535` the half stroke stays on the jacket and selects the
  jacket, as the click does.
- **A stroke across the whole region selects the intended mask without a level change on 23 of
  25 prompts** (both models, 5 points along the whole inner chord on the long axis of the region).
  With 2 or 4 points, base plus reaches all 25. The stroke tells the model the extent of the
  region: the person, the bassist, the kneeling woman, the squirrel, and the car with the kayak
  are selected as whole objects, and parts (hair, jacket, face, tail) stay parts. The stroke
  length is the main factor: with 5 points, 15 targets at a quarter of the chord, 17 to 18 at
  half, and 23 at the whole chord (section 7.2).
- **Few points are better than many.** 2 to 4 points are best; 8 points lose 2 to 4 targets
  (all-direction share 0.66 to 0.69 against 0.80 to 0.87 at the whole chord). Dense points on a
  small region fail: the lupine spike (a 23 px stroke) gives 0.31 with 5 points and 0.58 with
  3 points (base plus), against 0.99 for one click.
- **Add and subtract strokes work with all points in one decoder run and the previous mask as
  `mask_input`** (the SAM refinement path). One click, then corrections: every target is reached
  within 3 interactions (base plus mean 1.2, tiny 1.4). A whole-chord stroke, then corrections:
  mean 1.08 (base plus) and 1.12 (tiny). From the largest mask of a click (usually the whole
  object), one subtract stroke reaches the part on 7 (base plus) and 8 (tiny) of the 14 part
  prompts, from IoU 0.07 to 0.57 to 0.93 to 0.99. Without `mask_input`, some corrections lower the
  IoU (base plus: 2 of 3 subtract strokes). Strokes decoded alone and combined as a union or a
  difference of masks leave 1 (base plus) and 3 (tiny) prompts below the target after 5
  interactions.
- **Decoder time per stroke: 10.5 ms only for the first point count of a DirectML session.** A
  DirectML session runs the point count of its first run at 10 to 11 ms. Every other point count
  takes 22 to 23 ms in a warm loop and 26 to 34 ms (median) in the mixed sequence of the report.
  The order does not matter: a session that runs 5 points first is fast for 5 points and slow for
  1 point. Profiles of new sessions show the same nodes on DirectML for 1, 2, and 5 points.
- **A fixed point count by padding is not possible.** Padding to 32 points with padding labels
  (-1) or with copies of the last point keeps 11 ms, but SAM 2.1 changes the mask: 0 of 25
  targets for one padded click (base plus). PyTorch gives the same padded masks (IoU 1.00000
  against ONNX), so this is the model, not the export.
- **One decoder session per point count keeps 12 to 13 ms** with the same masks (maximum IoU
  difference 0.0), at about 90 MB dedicated GPU memory and 178 ms load time for each new point
  count (18 to 19 sessions: +1.6 to 1.7 GB dedicated, +0.8 GB shared).
- **The ONNX graph matches PyTorch for strokes:** `StrokePromptLogitsMatchPytorchWithinTolerance`
  passed for tiny (390 of 390 decoder calls, minimum IoU 0.99996) and base plus (324 of 324,
  minimum IoU 0.99999), with up to 25 points, negative points, and `mask_input` (section 5).
- **Recommendation (section 9):** keep the click as the main entry. Add a stroke tool that
  decodes once when the stroke ends, with 3 to 4 points spread over the stroke, all strokes of the
  selection in one prompt, and the previous mask as `mask_input`; a modifier key for subtract
  strokes. The stroke must cross the region: a short stroke is a click. One decoder session at
  about 27 ms per stroke, or one session per point count at 12 ms and about 90 MB per count. Base
  plus (all 25 targets reachable with a stroke); tiny as the lower-memory alternative. The user
  decides (decision D2).

## 3. Interaction design

### 3.1 Stroke

1. The user drags a straight stroke inside the region. The host samples N points at equal
   spacing from the start to the end of the stroke (both ends included).
2. The point decoder runs once with the N points as positive points. Several points use the
   single-mask output (token 0, or the best multimask token when token 0 is not stable, the E1
   rule); there is no level to choose.
3. A stroke with a modifier key gives negative points (subtract stroke). A second stroke without
   the key adds positive points (add stroke).

### 3.2 Combining the interactions of one selection

| Combination | Decoder input for an add or subtract stroke |
| --- | --- |
| Points and previous mask (`points_and_previous_mask`) | All points of all interactions (positive and negative) and the low-resolution logits of the previous selection as `mask_input`. The interactive refinement of SAM. |
| Points (`points`) | All points of all interactions, no `mask_input`. |
| Union of strokes (`union_of_strokes`) | The new stroke alone, with positive points. An add stroke takes the union (maximum of the logits), a subtract stroke removes its mask (minimum with the negated logits). Each stroke is independent, like the brush of Photoshop. |

## 4. Method

### 4.1 Inputs

| Item | Value |
| --- | --- |
| Photos and prompts | The E4 input set (`build/tmp/explore/e4/input_4096/`): 14 photos scaled to the 4096 px application input, the 26 single-point prompts of plan section 8.15. |
| Intended selection | `scripts/ai_models/sam2/click_targets.json` (E4): 11 object and 14 part prompts with an E1 reference mask. `2088064441` person has no reference and is not measured. |
| Models | SAM 2.1 tiny and base plus, the E1 graphs (`point_encoder.onnx`, `point_decoder.onnx`; the copies in `build/tmp/explore/e4/onnx_float32_batch16/`, float32). |
| Scripts | `rust/puerh_mind/examples/segmentation_runtime_report.rs` (`strokes` mode) and [`scripts/ai_models/sam2/`](../../../../../scripts/ai_models/sam2/README.md): `compare_stroke_outputs.py`, `probe_strokes.py`, `make_stroke_sheets.py` |
| Evidence | `build/tmp/explore/e3/` on the Windows machine (not committed): `runtime/`, `masks/`, `compare/`, `probe/`, `profiles/` |

### 4.2 Machine

The machine of E1, E2, and E4: Windows 11 build 22635, Intel Core i7-12700H, 31.8 GB, NVIDIA GeForce
RTX 3080 Laptop GPU 8 GB. ONNX Runtime 1.24 through `ort` 2.0.0-rc.12 (production session
settings of `ort_runtime::load_session`, DirectML). PyTorch 2.14.1 (CPU, 4 threads) and
onnxruntime 1.31.0 (CPU) for the check. Only one measurement ran at a time. macOS Core ML is not
measured.

### 4.3 Simulated strokes

The strokes are synthetic. They model a user who strokes inside the intended region:

- **Region:** the E1 reference mask of the target at the 1024 px evaluation size (E4).
- **Margin:** every stroke point has a distance of at least 1 percent of the short photo side
  (about 7 px) from the edge of the region (exact Euclidean distance transform). If the click
  point is closer to the edge, the margin is half of its distance (minimum 1.5 px; the lupine
  spike).
- **Inner chord:** the longest segment through the click point, in the stroke direction, whose
  points all keep the margin.
- **Stroke:** a fraction of the inner chord (0.25, 0.5, 0.75, 1), centred on the click point as
  far as the chord allows. "Half chord" is 0.5, "whole chord" is 1. The median stroke length is
  14 percent (half chord) and 28 percent (whole chord) of the short photo side.
- **Direction:** the principal axis of the region component that holds the click point (the long
  axis), and the fixed directions 0, 45, 90, and 135 degrees.
- **Points:** a fixed count (2, 3, 4, 5, 8) or a fixed spacing (2, 4, 8 percent of the short side,
  at least 2 and at most 16 points).

Corrective interactions follow the click simulation of the SAM papers. After each interaction,
the error region with the deepest pixel (largest distance to its own edge) is either the missed
part of the reference (an add correction) or the extra part of the selection (a subtract
correction). A corrective click goes to that pixel. A corrective stroke goes through that pixel
along the principal axis of its component, over half of its inner chord, with 5 points and a
margin of half of the depth. A sequence stops at IoU 0.90 or after 5 interactions.

| Sequence | First interaction | Corrections | Combination |
| --- | --- | --- | --- |
| `click_then_clicks` | click, best predicted IoU (the E4 default) | clicks | points and previous mask |
| `click_then_clicks_without_mask_input` | click | clicks | points |
| `click_then_strokes` | click | strokes | points and previous mask |
| `stroke_then_strokes` | half-chord stroke, 5 points | strokes | points and previous mask |
| `stroke_then_strokes_without_mask_input` | half-chord stroke | strokes | points |
| `stroke_then_strokes_union` | half-chord stroke | strokes | union of strokes |
| `full_stroke_then_strokes` | whole-chord stroke, 5 points | strokes | points and previous mask |
| `largest_mask_click_then_clicks` | click, largest multimask output | clicks | points and previous mask |
| `largest_mask_click_then_strokes` | click, largest multimask output | strokes (mostly subtract) | points and previous mask |

The `largest_mask` sequences start from the whole object, so the part targets need subtract
corrections; the other sequences start mostly with a part and need add corrections.

### 4.4 Metrics

- **Reference IoU:** the selection (256 x 256 logits, bilinear, threshold 0) against the
  reference, both at the 1024 px evaluation size (E4).
- **Target reached:** reference IoU >= 0.85 (E4).
- **Interactions to the target:** the first interaction count with reference IoU >= 0.85.
- **Decoder time:** `Session::run` of the point decoder on DirectML, without the upsampling.
- **GPU memory:** peak `Dedicated Usage` and `Shared Usage` of the process
  (`run_with_gpu_memory.ps1`).

## 5. Agreement with PyTorch

`StrokePromptLogitsMatchPytorchWithinTolerance` (`compare_stroke_outputs.py`) replays every
corrective sequence of the DirectML report on both sides with the same preprocessed pixels:
`SAM2ImagePredictor.predict` and `point_decoder.onnx` on ONNX Runtime CPU. Each side feeds its own
previous selection back as `mask_input`. Requirement: IoU >= 0.99 at the photo size for every
decoder call. E1 did not test negative points with `mask_input` or more than 3 points.

| Model | Decoder calls passed | Minimum IoU | Maximum \|logit difference\| | Calls with `mask_input` | Calls with negative points | Most points in one call |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 390 / 390 | 0.99996 | 0.0002 | 91 | 65 | 25 |
| base plus | 324 / 324 | 0.99999 | 0.0015 | 58 | 37 | 25 |

**Padding (`PaddedPromptMatchesUnpaddedPytorch`, one photo, `DSC_1535`, the three
`stroke_then_strokes` decoder calls):**

| Model | Padding to 32 points | Padded PyTorch against unpadded PyTorch, median IoU | Padded ONNX against padded PyTorch, minimum IoU |
| --- | --- | ---: | ---: |
| base plus | label -1 | 0.62 | 0.99999 |
| base plus | copies of the last point | 0.58 | 0.99999 |
| tiny | label -1 | 0.38 | 0.99999 |
| tiny | copies of the last point | 0.74 | 1.00000 |

The padded masks are wrong in PyTorch as well. The decoder attends to every prompt token, so 31
extra tokens change the result.

## 6. Time and GPU memory (DirectML, RTX 3080 Laptop GPU)

### 6.1 Decoder time per point count

Warm p50 of 20 runs after 3 warm-up runs, without `mask_input`. "New session": a decoder session loaded for this point
count (its first input shape). "Report session": the session of the report, which ran the counts
in the order 1, 2, 3, 5, 8, 16, 32 before.

| Points | tiny, new session | tiny, report session | base plus, new session | base plus, report session |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.9 ms | 11.5 ms | 10.6 ms | 11.5 ms |
| 2 | 10.8 ms | 21.7 ms | 10.5 ms | 22.0 ms |
| 3 | 10.7 ms | 21.8 ms | 10.7 ms | 21.7 ms |
| 5 | 10.4 ms | 23.0 ms | 10.2 ms | 22.8 ms |
| 8 | 10.3 ms | 22.0 ms | 10.6 ms | 22.2 ms |
| 16 | 10.5 ms | 22.1 ms | 10.5 ms | 22.5 ms |
| 32 | 11.5 ms | 22.2 ms | 11.4 ms | 22.7 ms |

- `mask_input` does not change the time (within 1 ms in every row).
- The number of points does not change the time of a new session (10 to 12 ms for 1 to 32
  points).
- With 2 to 32 points, the report session takes 22 to 23 ms: about twice the time of a new
  session.
- A session that runs 5 points first, then 1, then 5 again: 11.0, 22.2, 10.4 ms. The first
  point count is fast, every later count is slow, in any order (trial, tiny).
- Profiles of new sessions with 1, 2, and 5 points (tiny, `--profile-dir`): the same 121 CPU and
  284 DirectML nodes, and the same three fused DirectML nodes at the top of the kernel time. The
  slow path of a second point count is not visible in the per-node profile of one shape.

Decoder time in the report (all decoder calls of the sweep and the sequences, mixed point
counts, median per point count): one session 16 to 17 ms for 1 point and 26 to 34 ms for 2 to
25 points; one session per point count 12 to 17 ms (one point count with a single call: 49 ms).

### 6.2 Ways to one input shape

| Way | Decoder time | Quality | Cost |
| --- | --- | --- | --- |
| Pad to 32 points with label -1 | 10.6 to 11.9 ms | tiny 0 to 1 of 25 first-interaction targets, base plus 0 to 1 | — |
| Pad to 32 points with copies of the last point | 10.6 to 12.0 ms | tiny 8 to 9 of 25, base plus 0 to 2 | — |
| One decoder session per point count | 12 to 13 ms (median in the report for the counts with more than 50 calls) | the same masks as one session (maximum IoU difference 0.0) | about 90 MB dedicated GPU memory and 178 ms load time per point count |

### 6.3 GPU memory

| Process | Peak dedicated | Peak shared |
| --- | ---: | ---: |
| tiny, encoder + one decoder session | 1,469 MB | 281 MB |
| base plus, encoder + one decoder session | 2,657 MB | 447 MB |
| tiny, one decoder session per point count (19 sessions) | 3,188 MB | 1,080 MB |
| base plus, one decoder session per point count (18 sessions) | 4,277 MB | 1,202 MB |

Strokes add no GPU memory over E1 and E4 (the same graphs). The report and the timing loop each
loaded one more decoder session at a time for the new-session column.

## 7. Quality

### 7.1 First interaction

25 prompts (11 object, 14 part). Stroke: principal axis, 5 points.

| Model | First interaction | Targets reached | Object | Part | Median IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| tiny | Click, best predicted IoU (E4 default) | 16 | 4 | 12 | 0.93 |
| tiny | Half-chord stroke | 17 | 6 | 11 | 0.92 |
| tiny | Whole-chord stroke | 23 | 11 | 12 | 0.94 |
| base plus | Click, best predicted IoU (E4 default) | 20 | 6 | 14 | 0.99 |
| base plus | Half-chord stroke | 18 | 5 | 13 | 0.96 |
| base plus | Whole-chord stroke | 23 | 10 | 13 | 0.98 |

The one-click values are the same as in E4 (same graphs and inputs). The multimask output of a
half-chord stroke with the best predicted IoU reaches 17 (tiny) and 20 (base plus); a stroke then
has the same ambiguity as a click.

### 7.2 Stroke length and point count

Principal axis: prompts reached of 25. In brackets: the share of all strokes (5 directions) that
reach the target.

| Model | Length (of the inner chord) | 2 points | 3 points | 4 points | 5 points | 8 points |
| --- | --- | --- | --- | --- | --- | --- |
| tiny | 0.25 | 15 (0.61) | 16 (0.64) | 16 (0.63) | 15 (0.56) | 10 (0.43) |
| tiny | 0.5 | 15 (0.64) | 18 (0.70) | 17 (0.68) | 17 (0.65) | 14 (0.45) |
| tiny | 0.75 | 21 (0.74) | 20 (0.79) | 22 (0.78) | 21 (0.75) | 17 (0.60) |
| tiny | 1 | 21 (0.77) | 22 (0.80) | 22 (0.78) | 23 (0.79) | 21 (0.66) |
| base plus | 0.25 | 18 (0.74) | 19 (0.75) | 18 (0.70) | 15 (0.64) | 13 (0.43) |
| base plus | 0.5 | 18 (0.75) | 19 (0.78) | 20 (0.80) | 18 (0.71) | 16 (0.58) |
| base plus | 0.75 | 23 (0.82) | 22 (0.84) | 22 (0.86) | 20 (0.80) | 16 (0.62) |
| base plus | 1 | **25 (0.84)** | 23 (0.86) | **25 (0.87)** | 23 (0.83) | 21 (0.69) |

- Longer strokes reach more targets in every column.
- 3 or 4 points give the highest share at each length. 8 points are worst at each length.
- Whole-chord failures (5 points): base plus the bassist (`P1024439` dress, 0.71: the guitar is
  not part of the reference and interrupts the region, so the inner chord ends at the guitar and
  the legs stay out of the mask) and the lupine spike (0.31); tiny the paddle board (0.79) and
  the lupine spike (0.14).

### 7.3 Point spacing

All strokes (5 directions, 4 lengths), spacing as a share of the short photo side:

| Model | Spacing | Points, median (maximum) | Share reached | Median IoU |
| --- | --- | --- | ---: | ---: |
| tiny | 0.02 | 6 (16) | 0.65 | 0.93 |
| tiny | 0.04 | 3 (16) | 0.66 | 0.94 |
| tiny | 0.08 | 2 (16) | 0.67 | 0.94 |
| base plus | 0.02 | 6 (16) | 0.71 | 0.94 |
| base plus | 0.04 | 3 (16) | 0.76 | 0.97 |
| base plus | 0.08 | 2 (16) | 0.81 | 0.98 |

A wide spacing (few points) is better than a dense one. A fixed count of 3 or 4 points spread over
the stroke is as good as the best spacing (section 7.2) and does not depend on the stroke length.

### 7.4 Stroke direction

Half chord, 5 points, prompts reached: tiny principal 17, 0 degrees 16, 45 degrees 16,
90 degrees 15, 135 degrees 17; base plus 18, 19, 17, 17, 18. The direction changes the result by
at most 2 prompts; the length matters more.

### 7.5 Per prompt

Reference IoU: one click (E4 default) / half-chord stroke / whole-chord stroke with 5 points /
whole-chord stroke with 3 points, and the interactions to the target from the largest mask with
subtract strokes (`largest_mask_click_then_strokes`; "—": not reached in 5).

| Photo | Prompt | Intent | tiny | base plus |
| --- | --- | --- | --- | --- |
| `DSC_1535` | torso | object | 0.53 / 0.54 / 0.99 / 0.99 / 1 | 0.54 / 0.54 / 0.99 / 0.99 / 1 |
| `DSC_1535` | hair | part | 0.96 / 0.92 / 0.94 / 0.97 / 3 | 0.99 / 0.97 / 0.98 / 0.98 / 3 |
| `DSC00326` | jacket | part | 0.96 / 0.94 / 0.93 / 0.93 / 2 | 1.00 / 0.99 / 0.98 / 0.99 / 2 |
| `DSC00326` | hair | part | 0.93 / 0.90 / 0.92 / 0.93 / 2 | 1.00 / 0.93 / 0.94 / 0.97 / 2 |
| `DSC00674` | face | part | 0.98 / 0.75 / 0.92 / 0.96 / 2 | 1.00 / 0.99 / 0.98 / 0.99 / 2 |
| `DSC00674` | hair | part | 0.91 / 0.94 / 0.92 / 0.95 / 2 | 1.00 / 0.99 / 0.97 / 0.97 / 4 |
| `DSC00936` | torso | object | 0.58 / 0.63 / 0.99 / 0.99 / 1 | 0.64 / 0.54 / 0.99 / 0.99 / 1 |
| `P1024439` | dress | object | 0.31 / 0.37 / 0.89 / 0.90 / 1 | 0.30 / 0.31 / 0.71 / 0.78 / 1 |
| `P1024439` | hair | part | 0.91 / 0.92 / 0.94 / 0.94 / 2 | 0.99 / 0.86 / 0.93 / 0.97 / 2 |
| `DSC_1188` | body | object | 0.58 / 0.96 / 0.98 / 0.98 / 1 | 0.99 / 0.59 / 0.99 / 0.99 / 1 |
| `DSC_1188` | tail | part | 0.96 / 0.96 / 0.96 / 0.97 / 2 | 1.00 / 0.98 / 0.98 / 0.97 / 2 |
| `DSC00527` | sky | part | 1.00 / 0.99 / 0.99 / 0.99 / 1 | 1.00 / 1.00 / 1.00 / 1.00 / 1 |
| `DSC00527` | tree | part | 0.95 / 0.95 / 0.89 / 0.89 / 1 | 1.00 / 0.96 / 0.96 / 0.96 / 2 |
| `DSC00527` | kneeling woman | object | 0.48 / 0.53 / 0.96 / 0.96 / 1 | 0.50 / 0.50 / 0.93 / 0.97 / 1 |
| `2088064441` | board | part | **0.82 / 0.75 / 0.79 / 0.81** / 2 | 0.98 / 0.94 / 0.89 / 0.90 / 1 |
| `DSC00728` | balloon | object | 0.97 / 0.88 / 0.97 / 0.97 / 1 | 0.98 / 0.94 / 0.98 / 0.98 / 1 |
| `DSC00728` | sky | part | 1.00 / 1.00 / 1.00 / 0.99 / 1 | 1.00 / 1.00 / 1.00 / 1.00 / 1 |
| `5761824892` | car | object | 0.01 / 0.81 / 0.94 / 0.98 / 2 | 0.01 / 0.98 / 0.97 / 0.98 / 1 |
| `5761824892` | kayak | part | 0.93 / 0.93 / 0.94 / 0.93 / 2 | 1.00 / 0.98 / 0.98 / 0.98 / 2 |
| `DSC00562` | duck | object | 0.97 / 0.95 / 0.96 / 0.97 / 2 | 0.99 / 0.98 / 0.98 / 0.99 / 2 |
| `DSC00602` | tram | object | 0.98 / 0.95 / 0.95 / 0.96 / 1 | 0.98 / 0.98 / 0.99 / 0.99 / 1 |
| `DSC_1456` | car | object | 0.03 / 0.98 / 0.98 / 0.97 / 1 | 0.99 / 0.84 / 0.99 / 0.99 / 1 |
| Nikon lupine | sky | part | 0.99 / 0.97 / 0.98 / 0.98 / 1 | 1.00 / 1.00 / 1.00 / 1.00 / 1 |
| Nikon lupine | rock | object | 0.89 / 0.87 / 0.86 / 0.85 / 3 | 0.98 / 0.94 / 0.94 / 0.96 / 3 |
| Nikon lupine | flowers | part | **0.13 / 0.14 / 0.14 / 0.28** / — | 0.99 / 0.31 / 0.31 / 0.58 / — |

- **The whole-chord stroke gives the whole object where the click gives a part:** the torso
  (jacket to person), the bassist, the kneeling woman, the squirrel (tiny), the car with the
  kayak (grille to car), and the car of `DSC_1456` (tiny).
- **A stroke on a small part can be worse than a click:** the lupine spike (base plus 0.99 with
  one click, 0.31 with a stroke whose 5 points lie in 23 px at the base of the spike), the face of
  `DSC00674` (tiny, half chord 0.75).
- **The half-chord stroke can select a part where the click selects the object** (base plus
  `DSC_1188` body 0.59: the stroke stays on the back of the squirrel).

### 7.6 Sheets

One row per prompt and model. Columns: one click (E4 default), half-chord stroke, whole-chord
stroke, the half-chord stroke after the corrections, the click after the corrections, and the
largest mask of the click after the (mostly subtract) strokes. Green: positive points and add
strokes; red: negative points and subtract strokes; yellow line: the edge of the reference. Under
each tile: the reference IoU and the interaction count.

**`DSC_1535`** (torso: the half-chord stroke stays on the jacket, the whole-chord stroke gives the
person; hair: subtract strokes remove the person from the largest mask):

![DSC_1535](images/e3_strokes/DSC_1535.jpg)

**`5761824892`** (car with the kayak: one click gives the grille, a stroke gives the car):

![5761824892](images/e3_strokes/5761824892.jpg)

**`P1024439`** (bassist: the whole-chord stroke ends at the guitar, base plus leaves out the legs):

![P1024439](images/e3_strokes/P1024439.jpg)

**Nikon lupine** (the spike: a stroke at the base of a small part fails; one click works with base
plus):

![Nikon lupine](images/e3_strokes/Nikon-D800e-raw-00007.jpg)

The other photos: [`DSC00326`](images/e3_strokes/DSC00326.jpg),
[`DSC00674`](images/e3_strokes/DSC00674.jpg), [`DSC00936`](images/e3_strokes/DSC00936.jpg),
[`DSC_1188`](images/e3_strokes/DSC_1188.jpg), [`DSC00527`](images/e3_strokes/DSC00527.jpg),
[`2088064441`](images/e3_strokes/2088064441.jpg), [`DSC00728`](images/e3_strokes/DSC00728.jpg),
[`DSC00562`](images/e3_strokes/DSC00562.jpg), [`DSC00602`](images/e3_strokes/DSC00602.jpg),
[`DSC_1456`](images/e3_strokes/DSC_1456.jpg).

## 8. Add and subtract strokes

Interactions to the target (IoU >= 0.85), at most 5 interactions:

| Model | Sequence | 1 | 2 | 3 | 4 | 5 | Not reached | Mean (not reached = 6) | Add corrections, median gain | Subtract corrections, median gain | Corrections that lowered the IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| tiny | click, then clicks | 16 | 8 | 1 | 0 | 0 | 0 | 1.40 | 9, +0.47 | 2, +0.06 | 0 |
| tiny | click, then clicks, no `mask_input` | 16 | 6 | 2 | 1 | 0 | 0 | 1.52 | 17, +0.38 | 2, −0.15 | 5 |
| tiny | click, then strokes | 16 | 8 | 0 | 1 | 0 | 0 | 1.44 | 11, +0.43 | 2, +0.04 | 0 |
| tiny | half-chord stroke, then strokes | 17 | 7 | 1 | 0 | 0 | 0 | 1.36 | 13, +0.17 | 2, +0.04 | 0 |
| tiny | half-chord stroke, then strokes, no `mask_input` | 17 | 4 | 2 | 2 | 0 | 0 | 1.56 | 23, +0.08 | 1, −0.10 | 9 |
| tiny | half-chord stroke, then strokes, union | 17 | 2 | 2 | 1 | 0 | 3 | 1.96 | 25, +0.06 | 6, +0.06 | 3 |
| tiny | **whole-chord stroke, then strokes** | **23** | 1 | 1 | 0 | 0 | 0 | **1.12** | 6, +0.05 | 4, +0.05 | 1 |
| tiny | largest mask, then clicks | 12 | 10 | 2 | 0 | 1 | 0 | 1.72 | 6, +0.22 | 14, +0.53 | 1 |
| tiny | largest mask, then strokes | 12 | 10 | 2 | 0 | 0 | 1 | 1.76 | 5, +0.15 | 17, +0.49 | 2 |
| base plus | click, then clicks | 20 | 5 | 0 | 0 | 0 | 0 | 1.20 | 5, +0.48 | 0 | 0 |
| base plus | click, then clicks, no `mask_input` | 20 | 4 | 1 | 0 | 0 | 0 | 1.24 | 6, +0.42 | 0 | 0 |
| base plus | click, then strokes | 20 | 5 | 0 | 0 | 0 | 0 | 1.20 | 6, +0.42 | 0 | 0 |
| base plus | half-chord stroke, then strokes | 18 | 7 | 0 | 0 | 0 | 0 | 1.28 | 10, +0.39 | 0 | 0 |
| base plus | half-chord stroke, then strokes, no `mask_input` | 18 | 4 | 3 | 0 | 0 | 0 | 1.40 | 15, +0.20 | 3, −0.15 | 5 |
| base plus | half-chord stroke, then strokes, union | 18 | 3 | 2 | 1 | 0 | 1 | 1.60 | 17, +0.09 | 0 | 0 |
| base plus | **whole-chord stroke, then strokes** | **23** | 2 | 0 | 0 | 0 | 0 | **1.08** | 3, +0.26 | 0 | 0 |
| base plus | largest mask, then clicks | 13 | 9 | 3 | 0 | 0 | 0 | 1.60 | 3, +0.29 | 12, +0.69 | 0 |
| base plus | largest mask, then strokes | 13 | 8 | 2 | 1 | 0 | 1 | 1.80 | 2, +0.09 | 17, +0.57 | 0 |

- **With all points and the previous mask, no correction lowered the IoU in base plus** (0 of 58
  corrections) and 4 of 91 in tiny. Without `mask_input`, 5 to 9 corrections lowered it.
- **Subtract strokes:** from the largest mask (the whole person, car, or squirrel), one subtract
  stroke reaches the part on 7 (base plus) and 8 (tiny) of the 14 part prompts; on 4 more, the
  largest mask is already the part (the sky, the tree). The hair of `DSC_1535` needs two subtract
  strokes. The lupine spike is not reached (the largest mask is the whole field; 4 corrections
  give 0.80 with base plus and 0.69 with tiny).
- **Corrective clicks and corrective strokes reach the same targets** (`click_then_clicks`
  against `click_then_strokes`). A stroke does not correct faster than a click in these
  simulations; the corrections go to the deepest error pixel, where one point is enough.
- **The union of independent strokes is the weakest combination**: an add stroke decoded alone
  selects its own object, not the missing part of the first selection.

## 9. Recommendation

| Item | Recommendation | Evidence |
| --- | --- | --- |
| Main entry | One click with the E4 design (best predicted IoU, a level change for the rest) | 20 of 25 targets by default (base plus); the stroke does not beat it on the first interaction unless it crosses the region |
| Stroke tool | A stroke across the region selects the region without a level change. Decode once when the stroke ends. | Whole-chord stroke: 23 of 25 (both models); half-chord stroke: 17 to 18 |
| Points per stroke | 3 or 4 points at equal spacing over the stroke, both ends included | Highest share at each length (section 7.2); 8 points lose 2 to 4 targets; dense points on a small part fail |
| Add and subtract | Add stroke without a key, subtract stroke with a modifier key; all points of the selection in one decoder run with the previous mask as `mask_input` | 0 of 58 corrections lowered the IoU (base plus), 4 of 91 (tiny); union and no-`mask_input` variants are worse (section 8) |
| Decoder sessions | One session (about 27 ms per stroke, mixed point counts) if the decoder runs once per stroke. One session per point count (12 ms, about 90 MB and 178 ms load per count) only if the decoder must run during the drag. No padding. | Section 6 |
| Model | base plus; tiny as the lower-memory alternative | base plus: 25 of 25 with a whole-chord stroke of 2 or 4 points; tiny: 22 to 23 |

The user decides the interaction set and the model (decision D2, step E6).

## 10. Open items and next steps

- **User review** of the sheets in section 7.6.
- **Real strokes.** The strokes are straight and placed with the reference mask at a margin of
  1 percent of the short side; real strokes are curved, shorter or longer, and can leave the
  region. A test with recorded user strokes in the application is not done.
- **Stroke during the drag.** A live update while the stroke grows (Photoshop) needs a decoder run
  per new point and, on DirectML, a session per point count (section 6.2) or 22 to 34 ms per run.
  The UI response is not measured.
- **The DirectML shape effect** (fast only for the first input shape of a session) is measured,
  not explained. Its cause in the DirectML execution provider of ONNX Runtime 1.24 is not
  investigated; a newer ONNX Runtime may behave differently.
- **macOS Core ML** is not measured.
- **Float16** (accepted by the user on 2026-10-19) is not measured for the point decoder with
  strokes.
- **Tiny failures:** the paddle board (0.79 to 0.82) and the lupine spike (0.14 to 0.28) are not
  reached with a stroke; one click with corrections reaches them.

## 11. Changed files

- `rust/puerh_mind/examples/segmentation_runtime_report.rs`: new `strokes` mode (stroke placement
  with an exact distance transform, sweeps of point count, spacing, length, and direction,
  corrective click and stroke sequences with three combinations, decoder time per point count in
  new and existing sessions, operator profiles, `--pad-points` with `--pad-mode`,
  `--decoder-sessions per-point-count`); `run_point_decoder_with_mask` for `mask_input`. No change
  to product code.
- `scripts/ai_models/sam2/`: new `compare_stroke_outputs.py`, `probe_strokes.py`,
  `make_stroke_sheets.py`; `README.md`.
- `docs/roadmap/alcedo_studio/edit/ai_mask_candidate/`: this report and the index.
- `docs/roadmap/alcedo_studio/edit/ai_segmentation_mask_master_plan.md`: E3 result in section
  8.16.4.
