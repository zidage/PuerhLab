# AI Segmentation Mask (EfficientSAM3) Master Plan

Date: 2026-10-19

Status: SM1 export and runtime proof of concept done on 2026-10-19 (#448). The SM1 quality review
failed: the EfficientSAM3 follow-up work is stopped, and the model and interaction set are under
re-evaluation (section 8.16, #458). The other phases wait for that decision.

Target release: Alcedo Studio 0.3.4. The AI Segmentation Mask is the main feature of that release.

Parent context:

- [Node-aware Pipeline Editing and Mask Authoring Master Plan](node_mask_editor_master_plan.md)
- [NM7 Viewer Mask Creation and Editing](node_mask_editor/phase_nm7_viewer_mask_creation_plan.md)
- [Brush Mask Architecture, History, and Raster Materialization Master Plan](brush_mask_architecture_master_plan.md)
- [AI Sidecar Backend Integration Plan](../ai/ai_sidecar_backend_plan.md)
- [AI Sidecar Runtime Residency Plan](../ai/ai_sidecar_runtime_residency_plan.md)
- [Semantic Generation and Search Integration Plan](../ai/semantic_generation_search_plan.md)

Other affected areas: AI sidecar (`rust/puerh_mind`), model download (`app/model_*`), editor
render requests, GPU mask passes on CUDA, OpenCL, and Metal, and the editor QML workspace.

Source revision used for the audit: `e1effd87` on `fix/color-wheel-tonal-lgg`.

---

## 1. Decision record

### 1.1 Product goal

The user selects an object in the editor viewport with an AI model. The model makes a Mask that
follows the object edges. The Mask joins the existing Mask system of a Color Grade node. It uses the
existing R8 coverage, Union, and Grade Mix behavior. The interaction must be visually rich, because
this feature is the main item of the 0.3.4 release.

### 1.2 Approved decisions

These decisions come from the user. An implementer must not change them without a new user
decision.

| ID | Decision |
| --- | --- |
| D1 | The 0.3.4 main feature is the AI Segmentation Mask. CFA-domain neural denoise and DAG node blending by luminance or color are not the main feature of this release. |
| D2 | The model family is EfficientSAM3. The product must support text prompts and point or box prompts, subject to the SM1 results. Under re-evaluation since 2026-10-19 (section 8.16, #458). |
| D3 | The interaction is visually rich. The exact interaction set is decided in the UI phase. |
| D4 | The installer does not contain model weights. Alcedo Studio supplies a built-in downloader and the inference runtime. The weights come from the user-owned Cloudflare R2 bucket or from a Hugging Face repository. |
| D5 | The model files are treated as subject to the Meta SAM License. Each published model directory contains a copy of that license. SAM 2.1 is not a replacement candidate. Changed on 2026-10-19 by the user: SAM 2.1 is a candidate of the re-evaluation (section 8.16, #458). |
| D6 | The AI Segmentation Mask reuses the current R8 Mask logic: one R8 coverage per Mask, Union by per-pixel maximum, and Grade Mix. |
| D7 | The model always reads the whole source image. It does not read the image after the document crop and rotation. A later crop or rotation change must not make an existing AI Mask incomplete. |
| D8 | The analysis render disables the document crop in the same way that the Geometry panel does. See the source finding in section 5.3: the current Geometry panel mode keeps the rotation. |
| D9 | A change to Lens correction or to RAW decode parameters can move image content relative to an existing AI Mask. Alcedo Studio does not correct the Mask. The AI Mask UI tells the user that such a change can misalign a Mask. The user deletes the Mask and makes a new one. |
| D10 | This release does not include a color picker. The viewport pointer only supplies the prompt coordinates for the model. The coordinate rule must also be usable by a future color picker. |
| D11 | The first phase is a feasibility and measurement phase. Later design decisions are made as the plan advances. |

### 1.3 Proposed decisions that need user approval

These items came from the planning discussion. They are not approved. Each item names the phase that
must close it.

| ID | Proposal | Closed by |
| --- | --- | --- |
| P1 | Run inference in the `alcedo_mind` sidecar through ONNX Runtime. Do not add ONNX Runtime to the C++ process. | SM1 measurement, then user decision |
| P2 | Store prompts and Mask coverage in normalized ReferenceSpace. | SM5 |
| P3 | Store the R8 coverage only over the bounding rectangle of the Mask content, plus a margin. The maximum axis is `kMaximumRasterMaskAxis` (4096). | SM5 |
| P4 | Refine edges once when the user commits the Mask. Store the refined R8. Do not refine during each render. | SM1 quality review, then SM7 |
| P5 | Draw the temporary prompt result in the QSG layer. Show committed coverage through a pipeline coverage preview request. | SM8 and SM9 |
| P6 | Store a geometry fingerprint with each AI Mask. Mark only the Masks whose fingerprint differs from the current Lens and RAW decode geometry. | SM5 |
| P7 | Use one persistent gRPC channel per segmentation session. Keep only the newest pending decode request while the pointer moves. | SM3 |
| P8 | Ship Luminance Range and Color Range Masks in 0.3.4, because the coverage preview of SM8 also serves them. | User decision before SM8 |

### 1.4 Rejected options

- Do not encode only the cropped and rotated image (rejected by D7).
- Do not set `reference_bounds` to the document crop rectangle (rejected by D7).
- Do not move crop and rotation to a post-processing stage of the pipeline. The current mask passes
  already follow geometry through `render_to_reference` (section 5.2).
- Do not export a full image and read one pixel from it to implement a future color picker. A
  future picker reads a small region through a pipeline readback request.
- Do not bundle model weights in the installer (D4).

---

## 2. Terms

| Term | Meaning |
| --- | --- |
| ReferenceSpace | The uncropped and unrotated source image after the sensor active area. Normalized ReferenceSpace is `[0, 1]` on each axis of `full_reference_extent`. |
| Analysis render | A pipeline render that makes an 8-bit RGB image for the model. It does not apply the document crop or rotation. |
| Image embedding | The output of the EfficientSAM3 image encoder for one analysis render. |
| Prompt | One user input to the model: a positive point, a negative point, a box, or a text phrase. |
| Candidate Mask | The low-resolution mask that the model returns for the current prompts. It is not part of the document. |
| Committed Mask | A Mask that the document owns. It has a stored R8 coverage. |
| Segmentation session | The sidecar state for one analysis render: the loaded models and one image embedding. |

---

## 3. Product design specification

This section states the required behavior. The UI phase adds the detailed visual design after D3 is
refined.

### 3.1 Creation flow

1. The user selects a Color Grade node.
2. The user selects the AI Mask tool.
3. If the model is not installed, the tool shows a download panel. The panel shows the model name,
   the download size, and the license name. The user starts the download from that panel.
4. When the model is installed, Alcedo Studio starts a segmentation session for the current image.
   The viewport shows a busy state until the image embedding exists.
5. The user adds prompts. Each prompt change produces a new Candidate Mask. The viewport shows the
   Candidate Mask over the photo.
6. The user commits the Candidate Mask. Alcedo Studio refines the edges, stores the R8 coverage, and
   adds one Mask to the selected Color Grade node in one history operation.
7. The user can cancel. Cancel leaves the document unchanged.

### 3.2 Coordinate behavior

- A pointer position maps to normalized ReferenceSpace through the current viewport geometry.
- The user can add prompts only inside the visible photo. The resulting Mask can extend outside the
  current crop.
- A later crop, rotation, zoom, or region-of-interest render shows the same Mask on the same image
  content.

### 3.3 Failure behavior

| Failure | Required result |
| --- | --- |
| The model is not installed. | The tool shows the download panel. It does not start a session. |
| The download fails or a file checksum is wrong. | The download panel shows the real error. The installed model state does not change. |
| The sidecar does not start. | The tool shows the start error. The document does not change. |
| The model load or inference fails. | The tool shows the runtime error. It keeps the current prompts. It does not select another model, device, or resolution. |
| The user leaves the image during a session. | The session ends. The document does not change. |
| A stored AI Mask object is missing or corrupt. | Rendering reports the real error. It does not render an empty Mask or rebuild the Mask through another path. |

### 3.4 Exclusions

- A color picker is excluded from this release (D10).
- CFA-domain neural denoise is excluded from this plan.
- Blend modes and blending between DAG nodes are excluded from this plan.
- Luminance Range and Color Range Masks are excluded until the user closes P8.
- Automatic Mask correction after a Lens or RAW decode change is excluded (D9).

---

## 4. Scope by module

| Module | Responsibility in this plan |
| --- | --- |
| `scripts/ai_models/efficientsam3/` (proposed) | Export the selected checkpoint to ONNX. Check numeric agreement with PyTorch. |
| `rust/puerh_mind` | Load the ONNX sessions. Keep segmentation sessions. Serve encode and decode requests. |
| `rust/puerh_mind/proto` | Define the segmentation RPC messages. |
| `alcedo_studio/src/sidecar_client`, `app/ai_sidecar_runtime_service` | Call the segmentation RPCs with a persistent channel. |
| `app/model_asset_catalog`, `app/model_download_service`, Rust `model_assets.rs` | Download and verify the segmentation model files. |
| `edit/geometry`, `renderer/` | Make the analysis render in ReferenceSpace. |
| `edit/mask` | Own the AI Mask source model, validation, serialization, and the R8 asset. |
| `edit/runtime/{cuda,opencl,metal}` | Sample a raster Mask in `MaskEvaluate`. Compose the coverage preview. |
| `app/editor_mask_creation_controller`, `ui/alcedo_main` | Own the AI Mask tool, prompts, and the viewport presentation. |

Related plans:

- This plan extends the Mask model of the Node-aware Mask master plan.
- This plan shares the raster Mask storage and the three-backend raster sampling with the Brush Mask
  master plan. SM5 and SM6 must decide which parts this plan builds first. The Brush plan keeps its
  tiled copy-on-write storage design.
- This plan extends the AI sidecar with a local inference task. It does not change remote provider
  behavior.

---

## 5. Current source audit

### 5.1 Mask model and execution

| Area | Current owner and path | Current behavior | Required change |
| --- | --- | --- | --- |
| Mask source | `include/edit/mask/mask_model.hpp` | `MaskSource` is a variant of `RadialMaskSource` and `LinearGradientMaskSource`. `BrushMaskSource` exists only with `ALCEDO_ENABLE_BRUSH_MASK`. Analytic parameters are in normalized ReferenceSpace. | Add an AI Mask source variant (SM5). |
| Range fields | `mask_model.hpp:104-141` | `ColorRangeModel` and `LuminanceRangeModel` hold only `enabled = false`. The loader rejects other values. | No change unless the user closes P8. |
| Raster asset | `include/edit/mask/mask_asset.hpp` | `MaskAsset` holds packed R8 pixels, an extent, and `reference_bounds`. `MakeMaskAssetKey` hashes canonical bytes with 128-bit xxHash. No production code stores or uploads a `MaskAsset`. | Add persistence (SM5) and GPU upload (SM6). |
| Mask compile | `edit/runtime/graph_compiler.cpp:266-311` | `CompileGradeOwnedMaskStack` adds one `MaskEvaluate` pass per Mask and one `MaskUnion` pass. Each `MaskEvaluate` already receives a `range` input from the Grade scene input. Brush throws "Brush mask rasterization is not available in this build". | Compile the AI Mask source to a raster `MaskEvaluate` (SM6). |
| Native evaluation | `edit/runtime/cuda/cuda_mask_pass.cu:40-120`, `opencl/opencl_mask_pass.cpp`, `metal/metal_mask_pass.mm` | `AnalyticMaskKernel` maps each render pixel to ReferenceSpace with `render_to_reference` and evaluates the analytic formula. No backend samples a raster. | Add a raster sampling path on all three backends (SM6). |
| Raster sampling plan | `include/edit/geometry/texture_sampling_plan.hpp` | `MakeRasterMaskSamplingPlan(geometry, reference_bounds, texture_extent)` returns `render_to_texture_uv`. Production masks do not use it yet. | Use it for the AI Mask (SM6). |
| Mask thumbnail | `app/mask_thumbnail_evaluate.cpp:69-130` | A CPU loop draws analytic Masks at 128 x 128. It ignores image content. | Draw the AI Mask from its R8 asset (SM9). |

### 5.2 Geometry and coordinates

| Area | Current owner and path | Current behavior | Required change |
| --- | --- | --- | --- |
| Pointer mapping | `ui/edit_viewer/mask_edit_geometry.cpp:284-318` | `MaskEditGeometry::MapItemToReference` maps an item point to ReferenceSpace pixels and normalized ReferenceSpace. It handles a region-of-interest frame. | Reuse for prompts (SM9). |
| Frame geometry | `include/edit/geometry/resolved_render_geometry.hpp:32-50` | `ResolvedRenderGeometry` carries the decoded, reference, edit, and render transforms. | No change. |
| Pass order | `graph_compiler.cpp:417-508` | Develop runs Lens, then `GeometryResample`, then color conversion. Mask passes run per Color Grade at `render_extent`. | No change. Crop and rotation stay upstream. |

### 5.3 Analysis render

| Area | Current owner and path | Current behavior | Required change |
| --- | --- | --- | --- |
| Document geometry use | `include/edit/geometry/render_request.hpp`, `graph_compiler.cpp:72-87` | `DocumentGeometryUse` has `ApplyCropAndRotation` and `RotatedUncroppedSource`. `RotatedUncroppedSource` keeps the document rotation and frames the bounding box of the rotated source. Only the Geometry panel uses it. | The analysis render needs the unrotated, uncropped source. SM4 adds that mode or proves an equivalent existing path. |
| Host readback | `renderer/pipeline_scheduler.cpp:182-198`, `include/edit/runtime/frame_presenter.hpp:34` | `require_host_output` downloads the final display output. Only THUMBNAIL and FULL_RES_EXPORT renders set it. | Use it for the analysis render (SM4). |

### 5.4 Sidecar and model download

| Area | Current owner and path | Current behavior | Required change |
| --- | --- | --- | --- |
| ONNX Runtime | `rust/puerh_mind/Cargo.toml`, `src/service/ort_runtime.rs` | `ort` 2.0.0-rc.12 with DirectML and CoreML. The `auto` device lists DirectML then CPU on Windows, and CoreML then CPU on macOS. CUDA is rejected. The C++ process has no ONNX Runtime. | SM1 measures operator placement on each execution provider. |
| Session lifetime | `src/service/ort_clip.rs`, `encoder_residency.rs` | Encoders load on first use and unload after an idle time. One mutex serializes one encoder. | Add segmentation residency (SM3). |
| Transport | `sidecar_client/client.cpp:57-58` | Plain gRPC over loopback TCP. Each RPC makes a new channel and stub. The client uses the default 4 MB receive limit. | Use a persistent channel for segmentation (SM3). |
| Model catalog | `include/app/model_asset_catalog.hpp`, Rust `model_assets.rs` | The catalog lists semantic (CLIP) profiles only. `BuildAssetUrl` makes `{endpoint}/{repo_id}/resolve/{revision}/{path}`. Assets have size and an optional SHA-256. | Add a segmentation profile kind and a per-profile download location (SM2). |
| Download | `app/model_download_service.cpp` | aria2c downloads a profile into a staging directory, verifies it, and promotes it. `StartDownload` accepts only semantic profiles. | Accept segmentation profiles (SM2). |

### 5.5 Project format

`include/edit/history/pipeline_history_format.hpp` states that loaders accept only the current
format identities. The current project metadata version is `0.10.0`, and the Pipeline document JSON
version is 7. A new Mask source kind changes the Pipeline document JSON. SM5 must state the format
change and the result for older projects before implementation.

### 5.6 External model facts

These facts come from public pages on 2026-10-19. SM1 confirms them from the downloaded files.

- `facebook/sam3` on Hugging Face requires manual access approval. The current downloader does not
  send Hugging Face credentials.
- `Simon7108528/EfficientSAM3` on Hugging Face does not require access approval. Its license tag is
  Apache-2.0. It publishes PyTorch checkpoints (`.pt`, `.pth`), not ONNX files.
- EfficientSAM3 distils the SAM3 image encoder. The full checkpoints have about 89 to 95 million
  parameters, but the student image encoders have 0.68 to 22.4 million. This difference suggests that
  the full checkpoints contain SAM3 components. SM1 must confirm this inference.
- The EfficientSAM3 repository states that ONNX export is supported and that CoreML export is not
  available.

---

## 6. Target architecture (initial)

This section shows the intended owners. Each later phase confirms or changes its part before
implementation.

```text
Viewport pointer (QML)
  -> EditorMaskCreationAdapter
  -> MaskEditGeometry::MapItemToReference          normalized ReferenceSpace
  -> AI Mask tool controller (app layer)
       ├─ start: analysis render (ReferenceSpace, no crop, no rotation)
       │    -> sidecar StartSegmentationSession(image) -> image embedding stays in the sidecar
       ├─ prompt change: sidecar DecodeSegmentation(session, prompts) -> Candidate Mask
       │    -> QSG presentation over the photo
       └─ commit: edge refinement -> MaskAsset (R8, reference_bounds)
            -> Color Grade owner AddMask operation -> one history operation
Render
  -> CompileGradeOwnedMaskStack -> raster MaskEvaluate (CUDA / OpenCL / Metal)
  -> MaskUnion -> Grade Mix
```

The sidecar owns the model sessions and the image embedding. The document owns the committed Mask.
The project storage owns the R8 bytes. No owner keeps a copy of another owner's state.

---

## 7. Phase summary

| Phase | Result | Main modules | Dependency | Expected diff | Status |
| --- | --- | --- | --- | ---: | --- |
| SM1 | Model export, runtime feasibility, latency, and quality evidence | `scripts/ai_models`, `rust/puerh_mind` | none | 900-1500 | proof of concept done; quality review failed, re-evaluation (#458) |
| SM2 | Segmentation model download from R2 or Hugging Face | model catalog, download service, settings | SM1 | 800-1400 | outline |
| SM3 | Sidecar segmentation service and C++ client | proto, Rust service, `sidecar_client`, app service | SM1 | 1400-2000, split if over | outline |
| SM4 | Analysis render in ReferenceSpace | geometry, scheduler, app service | none | 600-1200 | outline |
| SM5 | AI Mask source model, R8 persistence, history, and project format | `edit/mask`, history, storage | SM1 | over 2000, split before start | outline |
| SM6 | Raster `MaskEvaluate` on CUDA, OpenCL, and Metal | `edit/runtime` | SM5 | 1000-1800 | outline |
| SM7 | Commit: edge refinement, asset publication, misalignment notice | app layer, `edit/mask` | SM3, SM4, SM5 | 800-1500 | outline |
| SM8 | Coverage preview request | render intent, compiler, three backends | SM6 | 1000-1800 | outline |
| SM9 | AI Mask tool UI and interaction | QML, app controllers | SM2, SM3, SM7, SM8 | over 2000, split before start | outline |
| SM10 | Product qualification | all | SM1-SM9 | 300-800 | outline |

"Outline" means that the phase has a scope and an estimate only. Each outline phase receives a
detailed phase plan with the alcedo-create-plan skill before it starts. That plan records the user
decisions from section 1.3 that the phase closes.

---

## 8. Phase SM1 — Model export and runtime feasibility

### 8.1 Objective and deliverables

SM1 proves that one EfficientSAM3 checkpoint can run as ONNX in the `alcedo_mind` runtime. It
measures speed and quality on real photos. Its results decide P1 and fix the model inputs for SM2,
SM3, and SM7.

SM1 does not change the product. It adds export tooling, a measurement program, and a report.

Deliverables:

1. A checkpoint inspection report. It lists each tensor group, its parameter count, and its origin
   (EfficientSAM3 student or SAM3).
2. An ONNX export script with pinned Python dependencies.
3. ONNX files for each required graph, with input and output shapes.
4. A numeric agreement report between PyTorch and ONNX Runtime CPU.
5. A Rust measurement program in `alcedo_mind`. It uses the production session settings.
6. Latency and memory measurements on DirectML (Windows) and CoreML (macOS).
7. An operator placement report for each execution provider.
8. A quality report on a fixed set of real photos, with user-confirmed visual review.
9. An edge refinement prototype comparison.
10. A gRPC round-trip measurement for a persistent channel and for a new channel per call.
11. A list of the files to publish, with sizes, SHA-256 values, and license files.
12. A recommendation for each item in section 8.13.

### 8.2 Inputs and prerequisites

- Read `AGENTS.md` again before you start. Read the alcedo-msvc-cmake skill for any C++ build.
- Download the EfficientSAM3 checkpoints from `Simon7108528/EfficientSAM3`. Do not use
  `facebook/sam3` unless the inspection proves that EfficientSAM3 needs it. In that case, stop and
  report (section 8.12).
- Clone `SimonZeng7108/efficientsam3` at a fixed commit. Record the commit hash.
- Use a Windows machine with a DirectML-capable GPU and a macOS machine with Apple silicon. Record
  the GPU, driver, OS, and memory of each machine.
- Prepare 12 real photos as listed in section 8.6. Use photos that the user owns.

### 8.3 Files

Proposed files:

| Path | Responsibility |
| --- | --- |
| `scripts/ai_models/efficientsam3/requirements.txt` | Pinned Python packages for inspection and export. |
| `scripts/ai_models/efficientsam3/inspect_checkpoint.py` | Write the tensor group report for one checkpoint. |
| `scripts/ai_models/efficientsam3/export_onnx.py` | Export the required graphs to ONNX. |
| `scripts/ai_models/efficientsam3/compare_onnx_outputs.py` | Compare PyTorch and ONNX Runtime CPU outputs on fixed inputs. |
| `scripts/ai_models/efficientsam3/refine_edges_prototype.py` | Make the edge refinement comparison images. |
| `scripts/ai_models/efficientsam3/README.md` | State the source commit, the license files, and the commands. |
| `rust/puerh_mind/examples/segmentation_runtime_report.rs` | Load the ONNX graphs with `ort_runtime` settings and write latency, memory, and operator placement data. |

Current files that change:

| Path | Change |
| --- | --- |
| `rust/puerh_mind/src/service/ort_runtime.rs` | Make the session builder reachable from the example, if it is not reachable now. Do not change its behavior. |
| `rust/puerh_mind/Cargo.toml` | Register the example, if Cargo does not find it automatically. |

SM1 does not change C++ product code. The gRPC measurement in step 10 uses the existing
`HealthService.Ping` RPC and the existing client code in a measurement program under `build/tmp/`.
If a tracked C++ file is necessary, record the reason in the completion record.

Weights, ONNX files, photos, and reports are not committed. They stay under `build/tmp/sm1/`.

### 8.4 Data rules and invariants

- The model input is the whole image. The exporter and the measurement program never crop the image
  to a subject.
- SM1 result: the EfficientSAM3 preprocessing (`Sam3Processor.set_image`) stretches the whole image
  to 1008 x 1008. It does not keep the aspect ratio and does not pad. The exporter and the
  measurement program use the same stretch. The 1008 size is fixed by the precomputed position
  encoding of the checkpoints.
- Prompt coordinates enter the model in model-input pixels. The conversion is: normalized
  ReferenceSpace multiplied by 1008 on each axis.
- The measurement program uses the same graph optimization level and execution provider list as
  `ort_runtime::load_session`. A measurement with other settings is labeled as such.
- An operator that ONNX Runtime places on the CPU provider under the DirectML or CoreML device is
  reported. It is not hidden. The user decides whether that placement is acceptable.
- A skipped or unavailable measurement is reported as unavailable. It is not reported as zero.

### 8.5 Implementation steps

1. Create a Python environment under `build/tmp/sm1/venv`. Install
   `scripts/ai_models/efficientsam3/requirements.txt` and the EfficientSAM3 repository at the fixed
   commit.
2. Run `inspect_checkpoint.py` on the candidate checkpoints. Start with
   `efficientsam3_ft/efficientsam3_efficientvit.pt`, `efficientsam3_repvit.pt`, and
   `efficientsam3_tinyvit.pt`. For each file, report the tensor groups: image encoder, prompt
   encoder, mask decoder, detector, text encoder. Report the parameter count of each group. Report
   whether the file loads without a SAM3 checkpoint.
3. Run the EfficientSAM3 PyTorch example on two photos with a point prompt and a text prompt. Save the
   output masks. These outputs are the PyTorch reference for step 5.
4. Run `export_onnx.py`. Export these graphs separately:
   - the image encoder: image tensor in, embedding tensors out;
   - the point and box decoder: embeddings and prompt tensors in, mask logits and scores out;
   - the text path: text tokens in, text features out, and the detection head if text prompts need
     it.
   Record the opset, the dynamic axes, and the file size of each graph. If a graph does not export,
   record the exporter error and the failing operator.
5. Run `compare_onnx_outputs.py`. Compare PyTorch outputs with ONNX Runtime CPU outputs on the same
   inputs. Report the maximum absolute difference of the mask logits and the IoU of the thresholded
   masks. The required agreement is IoU of at least 0.99 for every compared mask.
6. Write `segmentation_runtime_report.rs`. For each graph and each device (`cpu`, `directml` on
   Windows, `coreml` on macOS), measure:
   - session load time;
   - first inference time after load;
   - warm inference time over 30 runs for the encoder and 100 runs for the decoder, with p50, p95,
     and maximum;
   - process memory after load and after inference;
   - the operators that run on each execution provider, from the ONNX Runtime profile or verbose
     log.
7. Run the measurement program in release mode on both machines. Save the JSON output under
   `build/tmp/sm1/runtime/`.
8. Run the quality set in section 8.6. Save each Candidate Mask as a PNG overlay on the photo.
9. Run `refine_edges_prototype.py`. For each quality photo, compare two results at the photo's long
   edge capped at 4096: (a) bilinear upsampling of the model mask, and (b) guided upsampling with the
   photo as the guide. Save side-by-side crops of hair, fur, foliage, and hard edges.
10. Measure the gRPC round trip with `HealthService.Ping`: 1000 calls with one persistent channel and
    1000 calls with a new channel per call. Report p50, p95, and maximum.
11. Compute the SHA-256 and size of each file to publish. List the license files: the SAM License
    and the EfficientSAM3 Apache-2.0 license.
12. Write the SM1 completion record and the recommendations in section 8.13.

### 8.6 Quality photo set

Use 12 photos. Export each one from Alcedo Studio with the default edit, no crop, and no rotation.

| Case | Required content |
| --- | --- |
| 1-2 | A portrait with hair against a busy background. |
| 3 | An animal with fur. |
| 4 | Sky with a complex horizon of trees or buildings. |
| 5-6 | A small subject that covers less than 5 percent of the frame. |
| 7 | A low-light photo with visible noise. |
| 8 | A photo of 40 megapixels or more. |
| 9 | A panorama with an aspect ratio of at least 3:1. |
| 10 | A portrait-orientation photo. |
| 11 | Several similar objects, for example several people. |
| 12 | A product or hard-edged object. |

For each photo, run fixed point prompts, one box prompt, and one text prompt. Store the prompts in a
JSON file so that the run can repeat.

For cases 5 and 6, also encode only a rectangle around the subject. Report the IoU between the
whole-image Mask and the rectangle Mask. This number shows the quality cost of D7 for small subjects.
It does not reopen D7.

### 8.7 Primary success call chain

```text
inspect_checkpoint.py -> tensor group report
  -> export_onnx.py -> encoder.onnx, decoder.onnx, text graphs
  -> compare_onnx_outputs.py -> IoU >= 0.99 on all compared masks
  -> segmentation_runtime_report (cpu, directml | coreml) -> latency, memory, operator placement
  -> quality run -> overlay images -> user visual review
  -> refine_edges_prototype.py -> edge comparison images -> user visual review
  -> completion record and recommendations
```

### 8.8 Primary failure and restore call chain

```text
export_onnx.py fails on an operator
  -> record the operator and the exporter error
  -> try a supported exporter option on the same graph (opset, dynamic axes)
  -> if the graph still fails: stop SM1 and report (section 8.12)

segmentation_runtime_report: session load fails on DirectML or CoreML
  -> record the ONNX Runtime error
  -> keep the CPU measurement as a separate result
  -> do not report the CPU result as a DirectML or CoreML result
```

SM1 changes no product state, so it has no state to restore.

### 8.9 Tests and evidence

| Name | Observable assertion |
| --- | --- |
| `OnnxMaskLogitsMatchPytorchWithinTolerance` | For every compared prompt, the thresholded ONNX mask and the PyTorch mask have IoU of at least 0.99. The script prints the maximum absolute logit difference. |
| `SegmentationGraphsLoadWithProductionSessionSettings` | Each ONNX graph loads through the `ort_runtime` session builder on `cpu` and on the platform GPU device, or the report records the real load error. |
| `DecoderOutputShapeMatchesExportedSpecification` | The decoder returns the mask and score shapes that the export report states. |
| `PingLatencyIsReportedForPersistentAndPerCallChannels` | The report contains p50, p95, and maximum for both channel modes. The phase does not require a fixed ratio. |

The two Rust checks run inside `segmentation_runtime_report` and write pass or fail to its JSON
output. They are not added to `cargo test`, because they need downloaded model files.

### 8.10 Build and run commands

```bash
# Python environment and export (Git Bash)
python -m venv build/tmp/sm1/venv
build/tmp/sm1/venv/Scripts/python -m pip install -r scripts/ai_models/efficientsam3/requirements.txt
build/tmp/sm1/venv/Scripts/python -I scripts/ai_models/efficientsam3/inspect_checkpoint.py \
  --checkpoint build/tmp/sm1/weights/efficientsam3_efficientvit.pt \
  --out build/tmp/sm1/inspect/efficientvit.json
build/tmp/sm1/venv/Scripts/python -I scripts/ai_models/efficientsam3/export_onnx.py \
  --checkpoint build/tmp/sm1/weights/efficientsam3_efficientvit.pt \
  --out-dir build/tmp/sm1/onnx/efficientvit
build/tmp/sm1/venv/Scripts/python -I scripts/ai_models/efficientsam3/compare_onnx_outputs.py \
  --checkpoint build/tmp/sm1/weights/efficientsam3_efficientvit.pt \
  --onnx-dir build/tmp/sm1/onnx/efficientvit --prompts build/tmp/sm1/prompts.json \
  --out build/tmp/sm1/compare/efficientvit.json
```

```powershell
# Rust measurement program (PowerShell, Windows)
Set-Location D:\Projects\pu-erh_lab\rust\puerh_mind
cmd /c script\cargo_msvc.cmd run --release --example segmentation_runtime_report -- `
  --onnx-dir ..\..\build\tmp\sm1\onnx\efficientvit --device directml `
  --photos ..\..\build\tmp\sm1\photos --prompts ..\..\build\tmp\sm1\prompts.json `
  --out ..\..\build\tmp\sm1\runtime\windows_directml.json
```

```bash
# Rust measurement program (macOS)
cd rust/puerh_mind
cargo run --release --example segmentation_runtime_report -- \
  --onnx-dir ../../build/tmp/sm1/onnx/efficientvit --device coreml \
  --photos ../../build/tmp/sm1/photos --prompts ../../build/tmp/sm1/prompts.json \
  --out ../../build/tmp/sm1/runtime/macos_coreml.json
```

Run `cargo test -- --skip live_` once after the `ort_runtime.rs` change. Report the pass count.

### 8.11 Exit criteria

- [x] The inspection report states for each candidate checkpoint whether it loads without a SAM3
      checkpoint, and which tensor groups come from SAM3.
- [x] The encoder and the point and box decoder export to ONNX, and
      `OnnxMaskLogitsMatchPytorchWithinTolerance` passes for them.
- [x] The text path either exports and passes the same comparison, or the report states the real
      blocking error.
- [x] The runtime report contains load, first-run, and warm latency (p50, p95, maximum) for each
      graph on `cpu`, `directml`, and `coreml`, or states which measurement is unavailable and why.
      Core ML detector warm latency has 3 runs, not 30 (section 8.15).
- [x] The operator placement report lists each operator that runs on the CPU provider under a GPU
      device. The Core ML detector placement is unavailable (section 8.15).
- [ ] The user reviewed the quality overlays and the edge comparison images. The completion record
      states the user result for each case. Open: #449.
- [x] The Ping report contains both channel distributions. Measured with the Rust client; the C++
      client measurement is open: #453.
- [x] The publish list contains file names, sizes, SHA-256 values, and license files. The
      Apache-2.0 text file is missing upstream: #454.
- [ ] Section 8.13 has a recommendation for each item, and the user has made a decision on P1.
      Recommendations are in #454; the user decisions are open.

### 8.12 Stop conditions

Stop SM1 and report to the user when:

- every EfficientSAM3 checkpoint needs the access-controlled `facebook/sam3` checkpoint to load;
- the point and box decoder cannot export to ONNX;
- the ONNX output fails the 0.99 IoU requirement and the cause is not found;
- DirectML or CoreML cannot load the encoder or decoder;
- the measurement needs a product change outside the files in section 8.3.

### 8.13 Decisions that SM1 prepares

SM1 gives evidence and one recommendation for each item. The user decides.

| Item | Evidence from SM1 |
| --- | --- |
| Checkpoint variant (EfficientViT, RepViT, or TinyViT; size S, M, or L) | Quality review, latency, file size. |
| Text prompt support in 0.3.4 | Text path export result, latency, quality. |
| P1: sidecar or C++ process | Decoder latency compared with Ping round trip, operator placement, process memory. |
| CoreML use on macOS | CoreML load result, operator placement, latency. |
| Acceptable CPU operator placement under DirectML and CoreML | Operator placement report. |
| P4: edge refinement method | Edge comparison images and user review. |
| Model input size | Export shapes and the small-subject IoU of cases 5 and 6. |
| Published file list | Publish list with sizes and SHA-256 values. |

### 8.14 Expected diff

900-1500 lines: Python scripts and README 500-800, Rust example 300-550, Rust visibility changes
under 50, plan completion record 100. Generated files and reports are not counted.

### 8.15 Completion record

**Phase / date / status:** SM1, 2026-10-19. Export and runtime proof of concept done. The user
review (#449) and the user decisions (#454) are open. Issue: #448.

**Source revision and branch:** `3c378d9c` on `feature/ai-segmentation-sm1-runtime-poc`.

**EfficientSAM3 repository commit:** `bd0936c788fed8d51fa799437f05abd97b401b06`.

**Checkpoints inspected** (`inspect_checkpoint.py`; every file loads with 0 missing and 0
unexpected keys, without a `facebook/sam3` checkpoint):

| File | Parameters | Point and box decoder | Text encoder |
| --- | ---: | --- | --- |
| `stage1_all_converted/efficient_sam3_efficientvit_m_geo_ft.pt` | 428.2M | yes (4.2M) | SAM3 (353.7M) |
| `stage1_all_converted/efficient_sam3_efficientvit_m.pt` | 428.2M | yes | SAM3 |
| `stage1_all_converted/efficient_sam3_repvit_m_geo_ft.pt` | not inspected separately; exported | yes | SAM3 |
| `stage1_all_converted/efficient_sam3_tinyvit_m_geo_ft.pt` | not inspected separately; exported | yes | SAM3 |
| `efficientsam3_ft/efficientsam3_efficientvit.pt` | 97.5M | no | MobileCLIP-S0 (42.5M) |
| `efficientsam3_ft/efficientsam3_repvit.pt` | 100.9M | no | MobileCLIP-S0 |
| `efficientsam3_ft/efficientsam3_tinyvit.pt` | 103.6M | no | MobileCLIP-S0 |

Tensor group origin: only the image encoder trunk is an EfficientSAM3 student (14.4M for
EfficientViT-M). The `geo_ft` and the plain stage 1 file differ only in the image encoder. The
detector transformer, geometry encoder, scoring, and segmentation head are byte-identical in the
stage 1 and stage 3 files. The stage 3 (`efficientsam3_ft`) files have no point and box decoder, so
point and box prompts need a stage 1 file. This confirms the inference in section 5.6: the full
checkpoints contain SAM3 components.

**Actual changed modules:**

- `scripts/ai_models/efficientsam3/`: `esam3_graphs.py` (shared graph wrappers, not in the
  section 8.3 list), `inspect_checkpoint.py`, `export_onnx.py`, `compare_onnx_outputs.py`,
  `refine_edges_prototype.py`, `requirements.txt`, `README.md`.
- `rust/puerh_mind/src/service/ort_runtime.rs`: items are `pub`; new `session_builder` returns the
  production builder and `load_session` commits it. No behavior change.
- `rust/puerh_mind/examples/segmentation_runtime_report.rs`: `runtime` and `ping` modes.

**Implemented behavior:** four ONNX graphs (`point_encoder`, `point_decoder`, `text_encoder`,
`detector_grounding`; tensor layout in the README), opset 17. The decoder returns the four mask
tokens; the host selects the masks as `MaskDecoder.forward` does. The Rust program reproduces the
model preprocessing, the prompt mapping, the mask selection, the bilinear upsampling, and the CLIP
BPE tokens, and compares its masks with the PyTorch masks.

**Explicitly unimplemented items:** user review of the quality set (#449); Ping with the C++ client, GPU memory, 40 MP photo (#453); a correct Core ML
configuration (#450); text path with one shared image embedding (#452).

**Primary success call chain:**

```text
export_onnx.py -> PointEncoder / PointDecoder / TextEncoder / DetectorGrounding -> torch.onnx.export
compare_onnx_outputs.py -> Sam3Processor.set_image + predict_inst | set_text_prompt (PyTorch)
                        -> onnxruntime CPU -> IoU and max |logit difference| -> reference_masks/*.png
segmentation_runtime_report runtime -> ort_runtime::load_session (production settings)
  -> preprocess (stretch to 1008, normalize) -> point_encoder -> point_decoder (prompts x 1008)
  -> select_masks -> upsample_to_mask -> compare_with_reference -> write_overlay
  -> tokenize -> text_encoder -> detector_grounding -> score > 0.5 -> masks
  -> profile_graphs (production settings + with_profiling) -> node count per provider -> JSON
```

**Primary failure and restore call chain:** a graph load failure is recorded in
`SegmentationGraphsLoadWithProductionSessionSettings` with the ONNX Runtime error, and the run
continues with the other graphs. A failed profiling pass is recorded per graph. SM1 changes no
product state.

**Build and test commands with exit codes:**

- `cargo_msvc.cmd build --release --example segmentation_runtime_report`: 0, no warnings.
- `cargo_msvc.cmd clippy --release --example segmentation_runtime_report`: 0; no warning in the
  changed files (existing warnings in other files).
- `cargo_msvc.cmd test -- --skip live_`: 0.
- macOS `cargo build --release --example segmentation_runtime_report`: 0.

**Discovered / passed / failed / skipped counts:** `cargo test`: 293 passed, 0 failed, 8 filtered
(`live_`). `OnnxMaskLogitsMatchPytorchWithinTolerance`: EfficientViT-M 33/33 (28 point and box,
5 text prompts), RepViT-M 28/28, TinyViT-M 28/28; minimum IoU 0.99994, maximum absolute logit
difference 0.0002. `DecoderOutputShapeMatchesExportedSpecification` and
`HostTokenizerMatchesPythonBpeTokens` (5/5 prompts) passed on every run.
`SegmentationGraphsLoadWithProductionSessionSettings` passed for all graphs on `cpu`, `directml`,
and `coreml`. Rust pipeline masks compared with the PyTorch masks (different resize code): median
IoU 0.997 to 0.998; 35 of 39 at 0.99 or more for EfficientViT-M (minimum 0.79, caused by a
different single-mask token selection on `child_center`), 28 of 28 for RepViT-M and TinyViT-M.

**Machines:**

- Windows: Intel Core i7-12700H, 31.8 GB, NVIDIA GeForce RTX 3080 Laptop GPU 8 GB, driver
  610.62, Windows 11 build 22635.
- macOS: Apple M4, 16 GB, macOS 27.0.1.
- ONNX Runtime 1.24 through `ort` 2.0.0-rc.12. PyTorch 2.14.1 (CPU), onnxruntime 1.31.0 (Python).

**Latency and memory results** (production settings unless stated; milliseconds, warm p50 / p95;
EfficientViT-M unless stated):

| Graph | Windows CPU | Windows DirectML | macOS CPU | macOS Core ML |
| --- | --- | --- | --- | --- |
| `point_encoder` | 739 / 768 | 47.7 / 92.5 | 600 / 624 | 372 / 382, wrong output |
| `point_decoder` | 38.8 / 44.3 | 12.6 / 14.0 | 18.2 / 18.3 | 74.2 / 78.6 |
| `text_encoder` | 3.8 / 4.3 | 1.0 / 1.2 | 1.9 / 1.9 | not measured (run stopped) |
| `detector_grounding` | 2667 / 2861 | 254.5 / 263.7 | 2383 / 2487 | about 11 900, NaN scores |

- Session load on DirectML: encoder 2.8 s, decoder 0.4 s, detector 2.4 s. Core ML detector load
  22.3 s.
- RepViT-M and TinyViT-M encoder on DirectML: 48.0 / 51.1 and 60.8 / 63.4. CPU: 748 and 978.
- Core ML MLProgram format (not production): encoder 322 / 326 with correct output; the decoder
  load stops the process (#450).
- Memory after inference: Windows DirectML working set 1.1 GB, private bytes 4.8 GB; Windows CPU
  working set 0.6 GB, peak 1.9 GB; macOS CPU peak 5.1 GB. GPU memory is unavailable (#453).

**Operator placement results:** DirectML runs `point_encoder`, `text_encoder`, and
`detector_grounding` completely (fused DirectML nodes, 0 CPU nodes). `point_decoder` places 121
nodes on the CPU provider (Gather 37, Unsqueeze 22, Sqrt 14, Concat 11, Slice 10, Cast 7, Div 7,
Mul 7, Reshape 3, Equal 1, OneHot 1, Where 1), 6.4 ms of kernel time in two runs. Core ML
(production): `point_encoder` 72 Core ML and 50 CPU nodes, `point_decoder` 56 and 220,
`text_encoder` 30 and 42. The Core ML detector profile did not finish in 30 minutes.

**Quality review (user result per case):** user result not recorded yet (#449). The quality set has
14 user photos in `build/tmp/sm1/quality/` (prompts in `quality/prompts.json`, 48 point and box
prompts, 28 text prompts). No panorama (case 9): the user removed it from the set on 2026-10-19.
Case 7 uses `DSC00674.jpg` (ISO 1250, visible grain at 100 percent) as a weak low-light substitute.
Two extra hard photos: `P1024439.jpg` (backlit hair, microphone stand and cable in front of the
subject) and `Nikon-D800e-raw-00007.jpg` (lupine field, mountain horizon).

| Case | Photo | Developer observation (EfficientViT-M; RepViT-M and TinyViT-M are similar unless stated) |
| --- | --- | --- |
| 1 | `DSC_1535.jpg` | Box and three points: whole person. One torso point: suit only. Text `person`: the subject and one pedestrian; `hair` correct. |
| 2 | `DSC00326.jpg` | Box: coarse edge and holes in the legs. The rectangle pass gives a clean body with legs. Text `person` and `hair` correct. |
| 3 | `DSC_1188.jpg` | Box and two points: whole squirrel. One body point misses the tail. Text `squirrel`: no detection (top score 0.03). |
| 4 | `DSC00527.jpg` | Sky point: whole sky, the tree horizon is coarse. Tree point: one tree. Text `sky` correct; `tree` no detection; `person` finds 1 of about 30 people. |
| 5 | `2088064441.jpg` | Box: person and board as one blob. Rectangle IoU (box) 0.661 EfficientViT-M, 0.663 RepViT-M, 0.635 TinyViT-M; the rectangle pass is clearly better. |
| 6 | `DSC00728.jpg` | Box: balloon with a wrong edge. Rectangle IoU (box) 0.800 EfficientViT-M, 0.791 RepViT-M, 0.826 TinyViT-M; the rectangle pass gives the correct outline. Sky point: sky with a gap above the mountains and a false patch on the cliff with grid artifacts. |
| 7 | `DSC00674.jpg` | Head-and-shoulders close-up. Box and points fail in all three variants: the face is left out and the mask has grid holes. Letterbox input fixes the box prompt. Text `person` and `hair` correct. |
| 8 | `5761824892.jpg` (45.6 MP) | Box: whole car. Kayak point correct. Text `car` correct; `kayak` and `tree` no detection. |
| 10 | `DSC00936.jpg` | Box and points: whole person; the edge is about 20 px outside the body, and the rectangle pass and the matte correct it. |
| 11 | `DSC00562.jpg` | One duck correct. Three duck points: ducks and the hedge behind them. Text `duck`: no detection; `building`: 10 buildings, correct. |
| 12 | `DSC00602.jpg` (42.3 MP) | Box: tram, ragged bottom edge near the shadow. Text `tram`: no detection (top score 0.23). |
| extra | `DSC_1456.jpg` | Car box and point correct. Text `car` correct; `person` finds 1 of many pedestrians. |
| extra | `P1024439.jpg` | Box and three points: bassist with the bass. Text `person`: both musicians. The rectangle pass loses part of the right arm. |
| extra | `Nikon-D800e-raw-00007.jpg` | Sky point: empty mask (EfficientViT-M), sky with grid artifacts in the clouds (RepViT-M), correct (TinyViT-M). Rock point: the whole lupine field. Text `sky` and `mountain` correct; `flower` no detection. |

Findings across the set:

- The point and box decoder often shows a regular grid of holes and islands (the 72 x 72 embedding
  grid) on large or low-contrast regions, in all three variants and on CPU and DirectML.
- Encoder preprocessing (`compare_preprocessing.py`): letterbox input (aspect ratio kept, gray
  padding) fixes the close-up portrait and the Nikon sky, but makes some single-point prompts worse
  (`DSC_1535` torso, car points). "Dual" (both encoder passes, the mask with the higher decoder IoU
  score) picks the better input in most prompts, but not for the close-up hair point.
- A second decoder pass with the selected logits as `mask_input` makes single-point masks smaller
  or wrong on this model. It is not recommended.
- The text path is the most reliable path for people, hair, sky, buildings, and mountains. It
  misses animals, ducks, flowers, trees, trams, kayaks, and most people in crowds at the 0.5 score
  threshold.

**Edge refinement review (user result):** user result not recorded yet (#449).
`refine_edges_prototype.py` compares (a) bilinear, (b) the gray guided filter (radius 8, eps 1e-3),
(c) a rectangle pass, and (d) a projection matte:

- (b) changes 0.00 to 1.04 percent of the mask pixels on the quality set; the change is not visible
  on hair or fur.
- (c) encodes a square of the source photo around the subject when the square covers less than half
  of the photo. It gives the largest quality gain for small and medium subjects (cases 2, 5, 6, 10).
  It does not change D7: the whole photo is still the first input.
- (d) projects each pixel color in a band of one logit cell around the (c) edge onto the line
  between the local mean colors of the sure foreground and the sure background, then smooths the
  result with a small color guided filter. It keeps single hairs and fur against smooth
  backgrounds (cases 3, 7, `P1024439`). It adds speckle on textured backgrounds such as asphalt
  (case 1, shoes). The Python prototype takes 0.04 to 12 s per mask on the CPU at 4096 pixels.

**Ping results** (Rust tonic client, 1000 calls, Windows loopback): persistent channel p50 0.20,
p95 0.32, maximum 0.60 ms; new channel per call p50 0.52, p95 13.4, maximum 26.6 ms.

**Publish list** (`point_decoder.onnx` is byte-identical for the three variants):

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| EfficientViT-M `point_encoder.onnx` | 85,471,721 | `f6f6f60b0794656a6524322389c6fa828b4c0cc998e566bf46a34aac2f99ccd1` |
| RepViT-M `point_encoder.onnx` | 98,952,875 | `fd47ab621081a29ee742cfaff287b7a6a75945ba6e4178e905be65ce12bcc90b` |
| TinyViT-M `point_encoder.onnx` | 117,556,605 | `ef28ac407b50f7ab45b1cff591c27dfcafbb8037becfcf71a568dbcf196b156f` |
| `point_decoder.onnx` | 22,302,046 | `201e25c652b342625fa6a2f7f48f744843f0edeb61217bcb316b3b2965abafc0` |
| EfficientViT `text_encoder.onnx` | 169,171,248 | `24325d3cf107ef585ba6d29d74e643355962bf2caf5d0e9c67ca7032318137a9` |
| EfficientViT `detector_grounding.onnx` | 201,571,788 | `d37a94b23eb5dbf7f8fd6daf47a833a8590b949e0e8ea06ad1d69910c14ab582` |
| SAM License (`sam3/LICENSE`) | 7,353 | `4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be` |
| Apache-2.0 text | missing upstream | #454 |

**Recommendations and user decisions:** recommendations are in #454. No user decision yet.

**Evidence path:** `build/tmp/sm1/` on the Windows machine: `inspect/`, `onnx/`, `compare*/`,
`runtime/`, `profiles*/`, `overlays/`, `refine/`, `variant_contact_sheet.png`, and `quality/`
(photos, prompts, runtime reports, overlays and contact sheets per variant, `preprocess/`, `refine/`,
`review.html`).

**Remaining defects or unavailable platforms:**

- Core ML gives wrong encoder output and NaN detector output with the production settings (#450).
- `ort_runtime` runs a session on the CPU provider without an error when the GPU provider fails to
  register (#451).
- The SM1 diff is about 2,550 lines (Rust example 1,406 after `rustfmt`, Python and README about
  1,100), more than the 900-1500 estimate of section 8.14.

### 8.16 Grid holes and model re-evaluation

**Date and status:** 2026-10-19. The user stopped the EfficientSAM3 follow-up work after the quality
review. Closed as not planned: #449, #450, #452, #453, #454. #451 stays open because it also affects
the CLIP encoders. The exploration of section 8.16.4 is tracked in #458. The user excluded the
SAM3 teacher image encoder from the exploration; SAM 2.1 and the task mask models are tested.

#### 8.16.1 Cause of the grid holes

The holes are also in the PyTorch output (the ONNX graphs match PyTorch, section 8.15). They come
from the student image encoder:

| Student trunk | Native output at a 1008 px input |
| --- | --- |
| EfficientViT-B1 | 256 x 32 x 32 |
| RepViT-M1.1 | 512 x 32 x 32 |
| TinyViT-11M | 1024 tokens (32 x 32) |

- The SAM3 teacher ViT (patch 14) gives a native 72 x 72 grid. `ImageStudentEncoder`
  (`sam3/model_builder.py`) resizes the 32 x 32 student output to 72 x 72 with bilinear
  interpolation. One student cell covers about 31.5 input pixels.
- Stage 1 distillation uses only MSE and cosine loss on the final 72 x 72 embedding
  (`stage1/configs/base_stage1.yaml`). The point and box decoder is never trained on student
  features.
- On difficult prompts the decoder logits stay near 0. The neck and the decoder upscale with
  transposed convolutions (kernel 2, stride 2, no overlap), which add a fixed pattern per cell. Near
  0, this pattern decides the sign of each pixel and makes the regular dots and holes.

Measurement (EfficientViT-M, PyTorch, 288 x 288 logits; script `build/tmp/sm1/probe/grid_probe.py`,
not committed):

| Prompt | Share of \|logit\| < 2 | Share of logit > 4 | Grid energy |
| --- | ---: | ---: | ---: |
| `DSC00674` box (close-up, fails) | 0.826 | 0.000 | 0.288 |
| `DSC00728` sky point (grid on the cliff) | 0.213 | 0.105 | 0.374 |
| `DSC00326` box (correct) | 0.012 | 0.019 | 0.031 |
| `DSC_1535` box (correct) | 0.049 | 0.089 | 0.022 |

Grid energy is the share of the high-pass logit spectrum on the lines of period 4 and 9 cells.

Mitigations that do not correct the mask:

- Four encoder passes with half-cell shifts, logits aligned and averaged: grid energy 0.29 to 0.16
  on `DSC00674`, but the hole and island area stays the same.
- A box filter on `high_res_feats_0` and `high_res_feats_1`: no improvement.
- Zero high-resolution features: the grid stays (grid energy 0.46), so it comes from the image
  embedding.
- A Gaussian blur of the logits (1.5 cells): no dots, but the mask is still wrong (the face is left
  out).
- A 2304 px input (native 72 x 72 student output): the face is included, but new holes appear.

Result: post-processing cannot correct the EfficientSAM3 point and box masks. The model is uncertain
on these regions.

#### 8.16.2 Prompt dependence

- One point is ambiguous (`DSC_1535` torso: suit only; `DSC_1188`: squirrel without the tail).
- The text path misses squirrels, ducks, flowers, trees, trams, kayaks, and most people in crowds at
  the 0.5 score threshold.
- The user must know which prompt type works for which subject. This is not acceptable for the 0.3.4
  main feature.

#### 8.16.3 Practice in other photo applications

This review does not use RapidRAW or similar new raw editors.

| Application | Main entry | Other objects | Edge stage |
| --- | --- | --- | --- |
| Lightroom | One click: Subject, Sky, Background, People with body parts (skin, hair, eyes, lips, teeth, clothes), Landscape | Objects (brush or box) | Feather and Edge controls |
| Capture One | One click: Subject, Background (inverted Subject) | AI Select | not documented |
| Photoshop | Select Subject (Device or Cloud quality) | Object Selection tool | Select and Mask |
| Apple Photos (Vision) | Subject lift: all foreground instances in one pass; a tap selects an instance (`VNInstanceMaskObservation`, `instanceAtPoint`) | not applicable | 512 x 512 network, then guided-filter upsampling to full resolution on the GPU; a confidence branch hides unreliable results |
| DaVinci Resolve | Magic Mask with Person and Object modes, Faster and Better quality | add and subtract strokes | consistency controls |

Common pattern:

- Task masks with dedicated models are the main entry. The user selects a task, not a prompt type.
- A click or box tool is the second entry. Candidates are computed before the click, so a click
  selects a candidate instead of asking the model to resolve an ambiguous point.
- The coarse mask and the edge quality are separate stages.
- A confidence check hides an unreliable mask instead of showing it.

#### 8.16.4 Exploration steps (#458)

Evidence rules: use the quality set and prompts of section 8.6 (`build/tmp/sm1/quality/`). Keep
weights, ONNX files, and reports under `build/tmp/explore/`; do not commit them. Run on the Windows
DirectML machine first; record macOS Core ML results when a candidate is selected. Record the
license of each model; a license that does not permit commercial use excludes the model.

| Step | Work | Question that it answers |
| --- | --- | --- |
| E1 | Run the point and box prompts of section 8.6 with SAM 2.1 (Hiera tiny, small, base plus, and large; 1024 px input). SAM 2.1 takes its high-resolution decoder features from the Hiera stages at stride 4 and 8, not from transposed convolutions of one low-resolution map. Record the probe metrics of section 8.16.1, the encoder and decoder latency and memory on DirectML, the file size, and the license. | Does an encoder with native multi-scale features remove the grid holes and the close-up failure? Which size is affordable? |
| E2 | Run dichotomous image segmentation and matting models on the 14 photos: BiRefNet lite, BiRefNet 1024, and the BiRefNet high-resolution and matting variants (MIT license). Measure latency, memory, file size, and the soft matte on hair and fur. Record photos with more than one subject (crowd, ducks). | Can one model supply the one-click Subject mask and the hair edge without a separate matte step? |
| E3 | Select and measure candidate models for Sky (semantic segmentation) and People body parts (human parsing: skin, hair, clothes, lips, eyes). | Which models give the Sky and People task masks with an acceptable license? |
| E4 | Compute instance candidates when the tool opens (the SAM 2.1 automatic mask generator with the E1 size, or a class-agnostic instance model). A click selects a candidate; the pointer position highlights it. Measure the candidate time and the single-point cases of section 8.16.2. | Does candidate selection remove the single-point ambiguity at an acceptable time? |
| E5 | Only if E2 is not sufficient: compare guided-filter upsampling of a soft matte (the Apple method) with the SM1 projection matte (section 8.15, method (d)). | Which edge stage is robust on textured backgrounds? |
| E6 | Write one table per task (Subject, Sky, People parts, object click): model, latency, file size, license, quality, recommendation. | The user decides the revision of D2 and the interaction set. SM2 and later phases are replanned after that decision. |

Reports of the steps are in [`ai_mask_candidate/`](ai_mask_candidate/README.md).

- E1 (2026-10-19, [report](ai_mask_candidate/e1_sam2_1_point_box_evaluation.md)): SAM 2.1
  removes the grid holes in all four sizes (largest grid energy 0.04 to 0.11, EfficientSAM3 0.54)
  and gives the whole person on the `DSC00674` close-up box. One point still selects a part (E4).
  DirectML encoder warm p50: tiny 76 ms, small 87 ms, base plus 161 ms, large 322 ms; decoder
  about 10.5 ms. Peak dedicated GPU memory 1.36 to 3.10 GB. Apache-2.0. Recommendation: tiny,
  with base plus as the quality alternative. The user review is open.

---

## 9. Outline phases

Each outline phase lists its scope and its open decisions. A detailed phase plan replaces the outline
before the phase starts.

### 9.1 SM2 — Segmentation model download

Scope:

- Add a segmentation profile kind to `model_asset_catalog` and to Rust `model_assets.rs`. Keep the two
  catalogs in agreement.
- Add a download location per profile, so that semantic profiles keep the current endpoint setting
  and the segmentation profile reads from R2 or a Hugging Face repository.
- Pin the SHA-256 of every segmentation file.
- Download the license files with the model files.
- Show the model state and the download action in the settings and in the AI Mask tool.

Open decisions: download location (R2, Hugging Face, or both), settings page placement.

### 9.2 SM3 — Sidecar segmentation service and C++ client

Scope:

- Add `segmentation.proto` with session start, decode, text prompt, and session end operations.
- Add the Rust service with model residency and an image embedding per session.
- Add the C++ client and the app service with a persistent channel and a receive limit that fits the
  largest decode response.
- Load the models when the user selects the AI Mask tool.

Open decisions: P1 and P7, the session lifetime, and the decode response format (logits or R8, and
its resolution). Split the phase into a Rust part and a C++ part if the estimate passes 2000 lines.

### 9.3 SM4 — Analysis render in ReferenceSpace

Scope:

- Add a render mode that applies no crop and no rotation, or prove that an existing mode gives the
  same frame. `RotatedUncroppedSource` keeps the rotation (section 5.3).
- Render the whole frame at the model input size, and a ReferenceSpace rectangle at up to 4096 on
  the long edge for edge refinement.
- Read the result back to the host with `require_host_output`.

Open decisions: which pipeline stages the analysis render applies (for example Develop and the
default display transform without Color Grade nodes), and the image embedding cache key.

### 9.4 SM5 — AI Mask source model, R8 persistence, history, and project format

Scope:

- Add the AI Mask source variant: the R8 asset key, the prompts, the model identity, and the
  geometry fingerprint if P6 is approved.
- Add project-owned R8 asset storage with verification on read.
- Integrate history, Undo, Redo, Version, Paste, reopen, and export.
- State the project format change and the result for older projects.

Open decisions: P2, P3, P6, the format change, and the storage shared with the Brush plan. Split this
phase before it starts. The expected diff is more than 2000 lines.

### 9.5 SM6 — Raster `MaskEvaluate` on three backends

Scope:

- Compile the AI Mask source to a raster `MaskEvaluate` pass.
- Sample the R8 texture with `MakeRasterMaskSamplingPlan` on CUDA, OpenCL, and Metal. Coverage
  outside `reference_bounds` is zero.
- Upload the R8 asset into the render workspace of the open image and release it when the image
  closes.
- Compare each backend with independent expected R8 values.

### 9.6 SM7 — Commit and misalignment notice

Scope:

- Refine the Candidate Mask with the method that the user selects in SM1.
- Make the `MaskAsset` and publish it with one Color Grade owner operation and one history
  operation.
- Show the misalignment notice of D9.

### 9.7 SM8 — Coverage preview request

Scope:

- Add a coverage preview flag to the editor render intent.
- Add a composite pass after the display transform that shows Mask coverage on the photo.
- Include the flag in the render cache key.

Open decisions: P5 and P8, and the visual style of the coverage.

### 9.8 SM9 — AI Mask tool UI and interaction

Scope:

- The AI Mask tool, prompt input, Candidate Mask presentation, commit, and cancel.
- The download panel, the busy state, and the error states of section 3.3.
- The Mask thumbnail from the R8 asset.

Open decisions: the exact interaction set of D3. Split this phase before it starts. The expected diff
is more than 2000 lines.

### 9.9 SM10 — Product qualification

Scope:

- Real RAW photos on CUDA, OpenCL, and Metal.
- Crop, rotation, zoom, region-of-interest render, and export with AI Masks.
- History, Version, Paste, reopen, and package.
- Installed-application download, license files, and a clean-machine start.
- Latency and memory compared with the targets that the user approves after SM1.

---

## 10. Cross-phase acceptance matrix

| Behavior | Expected result |
| --- | --- |
| Crop after Mask creation | The Mask covers the same image content. Content outside the old crop keeps its coverage. |
| Rotation after Mask creation | The Mask rotates with the image content. |
| Zoom and region-of-interest render | The Mask edge stays on the same image content at each zoom level. |
| Export | The exported Mask coverage matches the preview coverage on the same image content. |
| Undo and Redo of Mask creation | The Mask and its coverage appear and disappear with the history operation. |
| Reopen | A reopened project shows the same coverage. It does not run the model. |
| Missing model | The tool shows the download panel. The document does not change. |
| Sidecar failure | The tool shows the real error. The document does not change. |
| Corrupt R8 asset | Rendering reports the real error. It does not render an empty Mask. |
| Lens or RAW decode change | The Mask stays as stored. The UI shows the notice of D9. |

---

## 11. Build and evidence rules

- Use only the `win_debug`, `win_release`, `macos_debug`, and `macos_release` presets.
- Run MSVC commands through the PowerShell tool with `scripts\msvc_env.cmd`.
- Put all temporary files, weights, photos, and reports under `build/tmp/<phase>/`.
- Run only the test binaries for the changed code and their direct callers. Only the user starts a
  full test suite.
- Report pass, fail, skip, and unavailable results separately. Do not report an unavailable
  measurement as zero.
- Record user visual reviews as manual verification. Keep them separate from automated evidence.

---

## 12. Risks and stop conditions

| Risk | Detection signal | Required response |
| --- | --- | --- |
| EfficientSAM3 checkpoints need SAM3 weights. | SM1 step 2. | Stop and report. The user decides the model source. |
| DirectML or CoreML runs most operators on the CPU provider. | SM1 operator placement report. | Report the placement. The user decides. Do not change the device without a user decision. |
| Whole-image input is too coarse for small subjects. | SM1 cases 5 and 6. | Report the IoU. D7 stays. Propose a refinement method in SM7. |
| The edge quality is low at full resolution. | SM1 edge comparison. | Select the refinement method with the user before SM7. |
| The project format change rejects older projects. | SM5 format analysis. | Stop SM5 until the user approves the format change and its result. |
| The license terms change. | Meta SAM License page. | Report the change before SM2 publishes files. |

Stop implementation when:

- a required owner API does not exist;
- the source audit no longer matches the branch;
- a phase can pass the 2000-line limit;
- a new persistence format lacks a rejection or migration decision;
- an operation needs a fallback that the user did not authorize;
- a proposed consistency mechanism lacks a real production interleaving.

Before each phase, read `AGENTS.md` and each applicable skill again. Repository rules can change after
this plan was written.
