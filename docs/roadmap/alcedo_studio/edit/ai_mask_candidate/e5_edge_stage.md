# E5: Edge stage of the stroke selection (people)

**Issue:** #458, step E5. **Plan:** [AI Segmentation Mask master plan, section 8.16](../ai_segmentation_mask_master_plan.md#816-grid-holes-and-model-re-evaluation).
**Date:** 2026-10-20. **Branch:** `feature/ai-segmentation-sm1-quality-review`.
**Status:** measurements done. The user review of the sheets and the decision are open.

> **Images are local only.** The images under `images/` show photos that must not be shared
> again. They are in `.gitignore` and are not distributed with the GitHub repository. On a
> checkout without them, run the E5 command of the
> [script README](../../../../../scripts/ai_models/sam2/README.md#e5-edge-stage-of-the-stroke-selection)
> with the local evidence under `build/tmp/explore/`.

## 1. Question

The plan runs E5 only if E2 is not sufficient. E2 is not sufficient: BiRefNet lite gives a hard
hair edge and its speed is not usable (user assessment); the matting variants are dropped. E5
therefore compares the two edge stages of the plan on the selection that E3 recommends:

- **Guided-filter upsampling of a soft matte** (the Apple Vision method, plan section 8.16.3).
- **The SM1 projection matte** (plan section 8.15, method (d)).

User decisions for this step (2026-10-20): SAM 2.1 tiny is the model (no model matrix); the
selection tool is the stroke for now; the review looks at the edges of people. Step E5 answers:

1. Which edge stage is robust on textured backgrounds (asphalt, rock, bricks, leaves, water)?
2. What does the coarse selection need, so that an edge stage works?
3. What does the edge stage cost?

## 2. Result summary

- **The resolution of the coarse logits limits the edge, not the edge filter.** SAM 2.1 gives
  256 x 256 logits for the whole photo: one logit cell is 16 px at the 4096 px application input.
  At 16 px, neither edge stage is usable:
  - the guided filter spreads the edge over 30 to 45 px (median soft width 30.6 px on hair,
    34.6 px on hard edges). Where the background is flat, the filter output is the blurred matte,
    so a glow of 2 to 3 cells surrounds the person;
  - the projection matte breaks on textured and dark backgrounds: blocks of one logit cell and
    speckle at the shoes on the asphalt, the feet on the rock and on the bricks (561 small islands
    and holes in the 23 inspection regions), as in SM1.
- **A second encoder pass at about 4 px per logit cell makes both stages work.** Two forms were
  measured:
  - **Rectangle pass** (SM1 method (c)): one crop around the subject. It works only when the
    subject covers less than half of the photo: cells of 3.2 to 5.4 px for the small people,
    9.3 px for the bassist, and no gain for the two large people (`DSC_1535`, `DSC00674`).
  - **Tile pass** (new): 1024 px tiles along the edge of the photo selection, 4 px cells for any
    subject size, 2 to 19 tiles per person. Each tile gets the stroke points inside it, one
    positive and one negative point from the photo selection, and the photo logits as
    `mask_input`. The tile logits apply only within 2 photo cells (32 px) of the photo edge. The
    shoes, shoulders, hands, and feet then follow the image edges (median edge gradient on hard
    edges 2.50 to 3.23, photo pass 1.65 to 2.62).
- **The tile pass needs `mask_input` and the edge band.** Without `mask_input`, a tile decides
  the person again and loses parts (IoU against the photo selection 0.67 to 0.98; with it 0.95 to
  0.99). Without the edge band, a low-contrast region (the dark jacket of `DSC00674` against the
  dim sunset) gets opposite decisions in two neighbouring tiles, with a straight seam along the
  tile border. With an uncorrected coarse selection (`DSC00674` stopped at IoU 0.92, holes in the
  shoulder), whole tiles removed the shoulder completely.
- **On the tile pass, the two edge stages differ like this (23 regions, 6 people):**

  | Edge stage on the tile pass | Soft width, hair / hard (median) | Small islands and holes | Edge gradient, hard | Visual |
  | --- | --- | ---: | ---: | --- |
  | Guided-filter upsampling, radius 1 cell, eps 1e-3 | 9.5 / 10.1 px | 6 | 3.02 | Clean and smooth; curls and strand tips are blurred |
  | Guided-filter upsampling, radius 0.5 cell | 7.7 / 7.7 px | 0 | 2.85 | As above, a little sharper |
  | Projection matte, band 1 cell | 4.1 / 4.4 px | 116 | 3.23 | Sharpest on hands, feet, and shoes; some speckle remains on asphalt and in dark hair |

  **The guided filter is the robust stage on textured backgrounds** (0 to 6 specks against 116).
  The projection matte gives sharper edges but needs a speck cleanup.
- **Hair outside the coarse edge needs a wider band.** With a band of one tile cell (4 px), no
  stage on the tile pass brings back the curl tips that the coarse mask cut off (`DSC_1535`). The
  projection matte on the photo pass (band of 16 px) and on the tile pass with a band of 2 cells
  (8 px) recover part of the curls, at the cost of more specks.
- **The SAM logits are not a matte.** Inside the person the median logit is 5 and a quarter of
  the pixels are below 2.85 (rectangle pass: below 1.24), so `sigmoid(logits)` is 0.9 to 0.95
  inside. The guided filter on the sigmoid gives a gray, textured interior (soft width up to
  190 px). The soft matte must be the selection mask smoothed over half a cell (1 inside, 0
  outside), as a matting network output is.
- **Not solved by any variant:** single fly-away strands outside the hair mass (`DSC00674` top,
  the backlit strands of `P1024439`), and the dark jacket edge against the dim sunset
  (`DSC00674`, edge gradient 0.4 to 0.6 on the tile pass). These need a matting model or a
  trimap matting network; E5 does not test one.
- **Cost:** the tile pass is 2 to 19 encoder runs. With the E1 DirectML encoder time of tiny
  (76 ms warm p50), that is about 0.15 to 1.5 s per selection, plus one decoder run per tile. It
  must run once per committed selection (or when the pointer is idle), not per stroke. The edge
  stages are box filters and per-pixel arithmetic; the Python prototype takes 0.4 to 6 s on the
  CPU for the whole 4096 px photo. A GPU implementation and its time are not measured.
- **Recommendation (section 8):** for the stroke tool with SAM 2.1 tiny: the photo selection for
  the interaction, then at commit a tile pass along the edge and guided-filter upsampling of the
  smoothed mask (radius 0.5 to 1 tile cell, eps 1e-3). The projection matte is the option for
  sharper edges if a speck cleanup is added. The user decides (decision D2, step E6).

## 3. Pipeline

| Stage | What runs | Logit cell at 4096 px |
| --- | --- | --- |
| Selection | E3 practice: one stroke along the whole inner chord of the person, 4 points, single-mask output; corrective add and subtract strokes with all points and the previous logits as `mask_input` (E3, section 9) | 16 px |
| Coarse pass: photo | The 256 x 256 logits of the selection, bilinear to the photo size | 16 px |
| Coarse pass: rectangle | If the square around the selection covers less than half of the photo: the encoder on that crop of the 4096 px input; prompt: the stroke points inside it and the bounding box of the photo selection (SM1 method (c)) | crop side / 256 |
| Coarse pass: tiles | 1024 px tiles on a grid with 25 percent overlap, only the tiles within one photo cell of the selection edge. Prompt per tile: the stroke points inside it, a positive point at the deepest selected pixel, a negative point at the deepest unselected pixel, the photo logits of the tile as `mask_input`. The tile logits are blended with weights that fall to 0 over the overlap, and applied within 2 photo cells of the photo edge (a linear blend over one more cell) | 4 px |
| Edge: bilinear | Logits > 0 (the E3 and E4 masks) | — |
| Edge: guided | Soft matte: the cleaned mask (holes and islands below 3 x 3 cells removed) smoothed with a Gaussian of half a cell, on a grid of 4 px per cell. The color guided filter of He et al. with the photo as the guide on that grid; its coefficients applied at the photo size (fast guided filter). Radius 1 cell, eps 1e-3. Not thresholded | — |
| Edge: projection | The SM1 projection matte with its defaults: a band of one cell on each side of the edge; each band pixel color projected onto the line between the local mean colors of the sure foreground and the sure background; a color guided filter of radius 2 px | — |

The edge stages run on each coarse pass (9 main variants). Sensitivity variants on the tile
pass: guided on `sigmoid(logits)`, guided radius 0.5 and 2 cells, eps 1e-2 and 1e-4, projection
band 2 cells, whole tiles without the edge band, and whole tiles without `mask_input`.

## 4. Method

### 4.1 Inputs

| Item | Value |
| --- | --- |
| Photos | The E4 input set (`build/tmp/explore/e4/input_4096/`): the photos at the 4096 px application input. |
| People | [`edge_targets.json`](../../../../../scripts/ai_models/sam2/edge_targets.json): the six people of the quality set with a whole-person reference. The start point of the first stroke is the E3 click point where E3 has a person prompt (`DSC_1535`, `DSC00936`, `P1024439`, `DSC00527`), otherwise a point on the jacket (`DSC00326`) or the face (`DSC00674`). |
| Reference | The E1 PyTorch box mask of base plus at the E4 input size. It bounds the simulated strokes and gives the selection IoU. It is not an edge ground truth: no alpha ground truth exists for these photos. |
| Inspection regions | 23 crops chosen by eye, 256 to 512 px of the 4096 px input: 8 hair regions (soft edge expected) and 15 hard regions (skin or clothing against a textured or busy background; a sharp edge is expected). |
| Model | SAM 2.1 tiny, the E4 graphs (`build/tmp/explore/e4/onnx_float32_batch16/tiny/`, float32), ONNX Runtime 1.31 CPU. E3 showed that ONNX Runtime CPU and DirectML give the same stroke masks. |
| Scripts | [`refine_stroke_edges.py`](../../../../../scripts/ai_models/sam2/refine_stroke_edges.py) (the projection matte and the rectangle are imported from `scripts/ai_models/efficientsam3/refine_edges_prototype.py`), [`probe_edges.py`](../../../../../scripts/ai_models/sam2/probe_edges.py) |
| Evidence | `build/tmp/explore/e5/` on the Windows machine (not committed): `edges/` (`edges.json`, alpha PNGs of the main variants, logits), `probe/final.json`, `edges_stop090_whole_tiles/` (the earlier run with stop IoU 0.90 and whole tiles), `edges.log` |

### 4.2 Selection

- The first stroke and the corrective strokes are the E3 simulation, reimplemented in Python.
  `StrokePlacementMatchesE3`: for the four people whose start point and reference are E3
  prompts, the first stroke has the end points of the E3 sweep entry (4 points, principal axis,
  whole inner chord) within 0.70 px of the 1024 px evaluation frame.
- 4 points per stroke (E3 recommendation: 3 or 4).
- The sequence stops at selection IoU 0.95 or after 5 interactions. E3 used 0.90. E5 evaluates
  the edge, and a user corrects a visibly missing region (a shoulder) before the edge matters.
  The run with 0.90 is kept as evidence (section 6.3).

### 4.3 Machine

The machine of E1 to E4: Windows 11 build 22635, Intel Core i7-12700H, 31.8 GB, NVIDIA GeForce
RTX 3080 Laptop GPU 8 GB. Python 3 with OpenCV 5.0.0, NumPy 2.5.2, SciPy 1.18.1, 4 threads. All
times in this report are CPU times of the prototype. DirectML and macOS Core ML are not measured.

### 4.4 Metrics

No ground-truth alpha exists, so the metrics are proxies. The sheets are the main evidence.

- **Soft width:** pixels with 0.05 < alpha < 0.95, divided by the length of the alpha = 0.5
  contour, in the region. On hair a large value is expected. On a hard edge it is blur, a glow,
  or an alpha below 1 inside the person.
- **Specks:** small islands (alpha > 0.5) and holes (alpha <= 0.5) in the region, smaller than a
  quarter of a photo logit cell (64 px), not touching the crop border.
- **Edge gradient:** the mean photo gradient on the alpha = 0.5 contour, divided by the mean
  gradient in the band of one photo cell around the photo selection edge. Above 1: the contour
  lies on image edges.
- **Selection IoU:** at the 1024 px evaluation size against the reference.

## 5. Results

### 5.1 Selection and coarse passes

| Photo | Person | Interactions | Points (negative) | Selection IoU | Rectangle pass | Tiles | Photo against tiles, IoU | Without `mask_input` | Whole tiles |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `DSC_1535` | person, busy street | 1 | 4 (0) | 0.988 | not used (covers the photo) | 18 | 0.985 | 0.935 | 0.985 |
| `DSC00326` | person from behind, waterfall | 1 | 4 (0) | 0.977 | 17 % of the photo, 5.4 px cell | 7 | 0.970 | 0.921 | 0.970 |
| `DSC00674` | close-up portrait, dim | 2 | 8 (0) | 0.924, then 0.991 | not used | 19 | 0.980 | 0.698 | 0.960 |
| `DSC00936` | person against mist | 1 | 4 (0) | 0.985 | 16 %, 5.2 px | 6 | 0.982 | 0.977 | 0.982 |
| `P1024439` | bassist, busy background | 3 | 12 (4) | 0.892, 0.943, 0.951 | 45 %, 9.3 px | 8 | 0.956 | 0.671 | 0.956 |
| `DSC00527` | kneeling woman, bricks | 1 | 4 (0) | 0.961 | 6 %, 3.2 px | 2 | 0.952 | 0.864 | 0.952 |

- One whole-chord stroke selects five of the six people (the E3 result). `DSC00674` needs one
  add stroke on the shoulder. `P1024439` needs a subtract stroke (the bass guitar) and an add
  stroke.
- The tile pass keeps the selection (IoU 0.95 to 0.99 against the photo selection; selection
  IoU 0.952 to 0.991) and moves its edge. Without `mask_input`, tiles lose parts of the person.

### 5.2 Edge stages

All 6 people, 8 hair and 15 hard regions. The rectangle rows include `DSC_1535` and `DSC00674`,
where the rectangle pass is the photo pass.

| Coarse pass | Edge stage | Soft width, hair (median) | Soft width, hard (median / max) | Specks, hair | Specks, hard | Edge gradient, hair | Edge gradient, hard | CPU time, whole photo (median) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| photo | bilinear | 0 | 0 / 0 | 0 | 3 | 1.57 | 1.65 | 16 ms |
| photo | guided | 30.6 px | 34.6 / 44.8 px | 92 | 63 | 2.15 | 2.59 | 477 ms |
| photo | projection | 6.5 px | 6.3 / 15.6 px | 254 | 307 | 1.88 | 2.62 | 5.4 s |
| rectangle | bilinear | 0 | 0 / 0 | 1 | 81 | 1.75 | 1.46 | 6 ms |
| rectangle | guided | 19.0 px | 14.2 / 37.5 px | 83 | 43 | 2.24 | 2.21 | 500 ms |
| rectangle | projection | 6.2 px | 5.3 / 13.7 px | 197 | 238 | 2.02 | 2.35 | 1.7 s |
| tiles | bilinear | 0 | 0 / 0 | 0 | 64 | 1.98 | 2.50 | 15 ms |
| tiles | **guided** | 9.5 px | 10.1 / 11.5 px | **2** | **4** | 2.20 | 3.02 | 3.9 s |
| tiles | **projection** | **4.1 px** | **4.4 / 6.4 px** | 72 | 44 | 2.15 | **3.23** | 5.1 s |

The four people with a rectangle pass (`DSC00326`, `DSC00936`, `P1024439`, `DSC00527`):

| Coarse pass | Edge stage | Soft width, hair / hard (median) | Specks, hair / hard | Edge gradient, hair / hard |
| --- | --- | --- | --- | --- |
| photo | guided | 34.0 / 35.4 px | 9 / 26 | 3.05 / 3.15 |
| photo | projection | 6.0 / 6.3 px | 76 / 96 | 2.36 / 2.88 |
| rectangle | guided | 11.2 / 10.8 px | 0 / 6 | 2.91 / 2.12 |
| rectangle | projection | 4.4 / 5.3 px | 19 / 27 | 3.07 / 3.13 |
| tiles | guided | 8.7 / 9.2 px | 1 / 2 | 3.03 / 3.31 |
| tiles | projection | 4.1 / 4.4 px | 18 / 25 | 3.07 / 3.40 |

- **Photo pass:** the guided filter makes a glow of 30 to 45 px; the projection matte has the
  most specks (561) and blocks of one cell where the true edge is more than one cell away from
  the coarse edge (the top of the hair of `DSC00936` and `DSC00326`).
- **Rectangle pass:** close to the tile pass for the small people; worse for the bassist (9.3 px
  cell: guided soft width about 20 px) and no change for the large people.
- **Tile pass:** both stages follow the image edges (highest edge gradient on hard edges). The
  bilinear tile mask has dotted holes where the tile logits are near 0 (49 specks at the left
  shoe of `DSC_1535`); both edge stages remove them.

### 5.3 Per person

| Photo | Photo pass | Tile pass, guided | Tile pass, projection |
| --- | --- | --- | --- |
| `DSC_1535` | Bilinear cuts the curl tips; projection recovers part of the curls but breaks the shoes into specks; guided glows; holes at the ankle | Shoes, ankles, and shoulder follow the edge; curls cut to a smooth outline | As guided, sharper; curls not recovered (4 px band); 5 and 14 specks at the shoes |
| `DSC00326` | Projection blocks at the top of the hair; feet broken on the rock | Hands and feet clean; hair outline smooth | Wavy hair outline and fingers kept |
| `DSC00674` | Shoulder correct after the add stroke; hair edge soft | Hair outline smooth; shoulder edge low and ragged (dark on dark) | As guided; 19 specks in the dark hair on the right |
| `DSC00936` | Light line of mist along the arm (bilinear); glow (guided) | Clean | Clean, sharpest |
| `P1024439` | Stand and cable cut the arm; glow | Arm, hand, and legs follow the edge; backlit strands missing | As guided, sharper; 11 specks at the hand |
| `DSC00527` | Projection breaks at the feet on the bricks; the rectangle pass selects the gray bag with a dot pattern (uncertain logits) | Back clean, bag left out; feet smooth | As guided, sharper |

### 5.4 Sensitivity on the tile pass

| Variant | Soft width, hair / hard (median) | Specks, hair / hard | Edge gradient, hard |
| --- | --- | --- | ---: |
| guided, radius 1 cell, eps 1e-3 (main) | 9.5 / 10.1 px | 2 / 4 | 3.02 |
| guided, radius 0.5 cell | 7.7 / 7.7 px | 0 / 0 | 2.85 |
| guided, radius 2 cells | 12.6 / 12.8 px | 23 / 16 | 3.15 |
| guided, eps 1e-2 | 11.0 / 11.6 px | 0 / 1 | 3.01 |
| guided, eps 1e-4 | 8.2 / 8.5 px | 3 / 2 | 2.90 |
| guided on `sigmoid(logits)` | 28.5 / 22.5 px (max 189.5) | 3 / 5 | 2.98 |
| projection, band 1 cell (main) | 4.1 / 4.4 px | 72 / 44 | 3.23 |
| projection, band 2 cells | 5.2 / 5.2 px | 137 / 122 | 2.90 |
| whole tiles, bilinear | 0 / 0 | 15 / 92 | 2.50 |
| whole tiles without `mask_input`, bilinear | 0 / 0 | 203 / 116 | 2.45 |

- The guided radius and eps change the edge width by a few pixels; radius 0.5 to 1 cell is best.
  Visually the guided variants are alike (sheets `*__hair_settings.jpg`).
- A projection band of 2 cells recovers part of the curls of `DSC_1535` but doubles the specks.
- Guided on the sigmoid is not usable (gray interior).

### 5.5 Time (CPU prototype, whole 4096 px photo)

| Step | CPU time | DirectML estimate |
| --- | --- | --- |
| Encoder, photo pass | 1.4 to 3.1 s | 76 ms (E1, tiny, warm p50) |
| Encoder, rectangle pass | 1.5 to 1.7 s | 76 ms |
| Encoder, tile pass (2 to 19 tiles) | 3.2 to 31.2 s | 0.15 to 1.5 s (76 ms per tile) |
| Guided filter, tile pass | 3.7 to 4.3 s | not measured |
| Projection matte, tile pass | 4.9 to 5.8 s | not measured |

The prototype filters the whole photo; a production version filters only the band around the
edge. The DirectML estimate does not include the decoder runs (one per tile, 10 to 34 ms each,
E3) and the tile cropping.

### 5.6 Sheets

One sheet per person: the selection with its strokes (green: add, red: subtract), the rectangle
(cyan), and the tiles (violet); then per region the photo crop and, for each of the 9 main
variants, the photo with the background darkened by 2 EV and desaturated through the alpha (a
glow, a halo, or a missed strand is visible there) and the alpha.

**`DSC_1535`** (curly hair, shoulder against the street, shoes on the asphalt; the SM1 speckle
case):

![DSC_1535](images/e5_edges/DSC_1535.jpg)

**`DSC00674`** (close-up, fly-away hair, dark jacket against the dim sunset):

![DSC00674](images/e5_edges/DSC00674.jpg)

**`P1024439`** (backlit ponytail, microphone stand across the arm, busy background):

![P1024439](images/e5_edges/P1024439.jpg)

The other people: [`DSC00326`](images/e5_edges/DSC00326.jpg),
[`DSC00936`](images/e5_edges/DSC00936.jpg), [`DSC00527`](images/e5_edges/DSC00527.jpg).
The sensitivity variants on the hair regions:
[`DSC_1535`](images/e5_edges/DSC_1535__hair_settings.jpg),
[`DSC00326`](images/e5_edges/DSC00326__hair_settings.jpg),
[`DSC00674`](images/e5_edges/DSC00674__hair_settings.jpg) (whole tiles: the straight seam under
the right hair; without `mask_input`: the hair is lost),
[`DSC00936`](images/e5_edges/DSC00936__hair_settings.jpg),
[`P1024439`](images/e5_edges/P1024439__hair_settings.jpg),
[`DSC00527`](images/e5_edges/DSC00527__hair_settings.jpg).

## 6. Findings that changed the method

### 6.1 The soft matte

The first run used `sigmoid(logits)` as the soft matte of the guided filter. The interior of the
person was gray and textured (DSC00936: inside the person, 25 percent of the photo logits are
below 2.85 and of the rectangle logits below 1.24; the edge slope of the rectangle logits is 0.37
per cell). The SAM logits are a decision score, not an alpha. The main variant uses the cleaned
mask smoothed over half a cell instead.

### 6.2 The tile pass

The rectangle pass does not help the large people, so a tile pass was added after the first run.
A trial on `DSC_1535` (13 tiles) showed IoU 0.985 against the photo selection with `mask_input`
and 0.932 without it; the shoes and the shoulder then followed the image edge.

### 6.3 Whole tiles and the stop IoU

The first full tile run used whole tiles and the E3 stop IoU 0.90
(`build/tmp/explore/e5/edges_stop090_whole_tiles/`). `DSC00674` stopped after the first stroke at
IoU 0.924, with holes in the jacket. The tile pass then removed the shoulder: alpha 0 in the
shoulder region, IoU 0.869 against the photo selection (0.705 without `mask_input`). With stop
IoU 0.95, one add stroke fixes the jacket (IoU 0.991), but whole tiles still leave a straight
horizontal cut on the shoulder and a square notch under the right hair: two tiles decide the dark
area in opposite ways. Limiting the tile logits to 2 photo cells around the photo edge removes
both (IoU 0.960 to 0.980 against the photo selection). For the other five people, whole and
band-limited tiles give the same IoU.

## 7. Comparison with the plan question

| Question | Answer |
| --- | --- |
| Which edge stage is robust on textured backgrounds? | Guided-filter upsampling of the smoothed selection mask, on a tile pass: 6 specks in 23 regions (radius 0.5 cell: 0), no glow. The projection matte gives the sharpest edges (4.4 px on hard edges) but keeps 116 specks; on the photo pass it is the SM1 speckle (561). |
| What does the coarse selection need? | About 4 px per logit cell at the 4096 px input. The photo pass gives 16 px. A tile pass gives 4 px for any subject size; a rectangle pass only for subjects below half of the photo. The tile pass needs `mask_input` and must be limited to the edge band. |
| Cost | 2 to 19 tile encoder runs, about 0.15 to 1.5 s on DirectML (estimate from E1), once per committed selection. The filters on the GPU are not measured. |

## 8. Recommendation

| Item | Recommendation | Evidence |
| --- | --- | --- |
| Interaction | The stroke selection of E3 on the photo logits; the bilinear mask as the live preview | Section 5.1 |
| Coarse pass at commit | Tile pass: 1024 px tiles along the edge, the selection points, one positive and one negative point per tile, the photo logits as `mask_input`, tile logits only within 2 photo cells of the edge | Sections 5.1, 6.2, 6.3 |
| Edge stage | Guided-filter upsampling of the smoothed mask (Gaussian of half a cell), radius 0.5 to 1 tile cell, eps 1e-3, not thresholded | Fewest specks, no glow (sections 5.2, 5.4) |
| Option | The projection matte (band 1 cell) for sharper edges, with a speck cleanup (not tested); a band of 2 cells for curly hair | Sharpest edge; 116 specks (band 2: 259) |
| Not recommended | Any edge stage on the 16 px photo logits; guided on `sigmoid(logits)`; whole tiles; tiles without `mask_input` | Sections 5.2, 5.4, 6 |
| Fine strands | Not solved; needs a matting stage (a trimap matting network on the band of the tile pass, or a matting model) | Section 5.3 |

The user decides the edge stage and the interaction set (decision D2, step E6).

## 9. Open items and next steps

- **User review** of the sheets in section 5.6.
- **GPU implementation and time** of the tile pass (DirectML encoder per tile) and of the guided
  filter and the projection matte on the band only. The CPU times of the prototype do not
  represent the product.
- **Speck cleanup for the projection matte** (remove islands and holes below a quarter of a cell
  after the matte) is not tested.
- **Fine strands** (fly-away hair, backlit strands) are not captured. A trimap matting network on
  the tile band is not tested; the E2 matting models were dropped for speed.
- **Dark-on-dark edges** (`DSC00674` jacket against the dim sunset): the tile pass is worse than
  the photo pass there. A rule for low-contrast tiles (keep the photo logits) is not tested.
- **Real strokes.** The strokes are synthetic (E3); the edge band depends on the coarse edge,
  which depends on the strokes.
- **Other subjects.** E5 measures people only; fur (`DSC_1188`), foliage, and hard objects are not
  measured.
- **macOS Core ML** and **float16** are not measured.
- **Ground truth.** The metrics are proxies. A set with ground-truth alpha (a public matting set
  with a compatible license, or composites) would give an error measure.

## 10. Changed files

- `scripts/ai_models/sam2/`: new `edge_targets.json`, `refine_stroke_edges.py`, `probe_edges.py`;
  `README.md`. No change to product code.
- `docs/roadmap/alcedo_studio/edit/ai_mask_candidate/`: this report and the index.
- `docs/roadmap/alcedo_studio/edit/ai_segmentation_mask_master_plan.md`: E5 result in section
  8.16.4.
