# AI Segmentation Mask Master Plan

Date: 2026-10-19. Revised: 2026-10-22.

Status: SM1 complete on 2026-10-22. The EfficientSAM3 quality review failed (section 8.1). The
re-evaluation in the `alcedo-ai-mask-lab` repository accepted the model set SAM 2.1 Hiera tiny, the
people and sky head, and S3OD `s3od-synth` (section 8.2). Alcedo Studio does not use EfficientSAM3
or SAM3. SM3 is split into SM3.A, SM3.B, and SM3.C. SM3.A is complete on 2026-10-22
([phase plan](ai_segmentation_mask/phase_sm3a_segmentation_service_plan.md)); its macOS port check
ran with the SM3.B macOS run. SM3.B is complete on Windows and macOS on 2026-10-22
([phase plan](ai_segmentation_mask/phase_sm3b_embedding_cache_plan.md)). SM2 runs after SM9
(section 7).

Target release: Alcedo Studio 0.3.4. The AI Segmentation Mask is the main feature of that release.

Parent context:

- [Node-aware Pipeline Editing and Mask Authoring Master Plan](node_mask_editor_master_plan.md)
- [NM7 Viewer Mask Creation and Editing](node_mask_editor/phase_nm7_viewer_mask_creation_plan.md)
- [Brush Mask Architecture, History, and Raster Materialization Master Plan](brush_mask_architecture_master_plan.md)
- [AI Sidecar Backend Integration Plan](../ai/ai_sidecar_backend_plan.md)
- [AI Sidecar Runtime Residency Plan](../ai/ai_sidecar_runtime_residency_plan.md)
- [Semantic Generation and Search Integration Plan](../ai/semantic_generation_search_plan.md)
- [Alcedo AI Mask Lab](https://github.com/zidage/alcedo-ai-mask-lab): the model re-evaluation,
  the accepted model set, the export scripts, and a Python MVP of the AI Mask tool (section 8.2).

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
| D2 | Changed on 2026-10-22 by the user. The model set is SAM 2.1 Hiera tiny (stroke, click, box, add and subtract, and the final masks of Sky and People), the people and sky head on the SAM 2.1 image embedding (Sky and People localization), and S3OD `s3od-synth` (Subject; Background is the inverted Subject). EfficientSAM3 and SAM3 are not used. No model of the set takes text, so text prompts are not part of this plan. Evidence: section 8.2. |
| D3 | The interaction is visually rich. The exact interaction set is decided in the UI phase. The Python MVP of `alcedo-ai-mask-lab` is a reference for SM9, not a specification: SM9 decides each change to it (D15). |
| D4 | The installer does not contain model weights. Alcedo Studio supplies a built-in downloader and the inference runtime. The weights come from the user-owned Cloudflare R2 bucket or from a Hugging Face repository. |
| D5 | Changed on 2026-10-22 with D2. Each published model directory contains the license text of its models: Apache-2.0 for SAM 2.1, MIT and the DINOv3 License for S3OD. The people and sky head is an own model (training data COCO 2017 and Open Images V7). Model licenses must permit commercial use (lab decision U2). |
| D6 | The AI Segmentation Mask reuses the current R8 Mask logic: one R8 coverage per Mask, Union by per-pixel maximum, and Grade Mix. |
| D7 | The model always reads the whole source image. It does not read the image after the document crop and rotation. A later crop or rotation change must not make an existing AI Mask incomplete. |
| D8 | The analysis render disables the document crop in the same way that the Geometry panel does. See the source finding in section 5.3: the current Geometry panel mode keeps the rotation. |
| D9 | A change to Lens correction or to RAW decode parameters can move image content relative to an existing AI Mask. Alcedo Studio does not correct the Mask. The AI Mask UI tells the user that such a change can misalign a Mask. The user deletes the Mask and makes a new one. |
| D10 | This release does not include a color picker. The viewport pointer only supplies the prompt coordinates for the model. The coordinate rule must also be usable by a future color picker. |
| D11 | The first phase is a feasibility and measurement phase. Later design decisions are made as the plan advances. |
| D12 | 2026-10-22. The `alcedo_mind` sidecar runs the models, as it runs the CLIP encoders. Windows uses ONNX Runtime with DirectML. macOS uses compiled native Core ML models (`.mlmodelc`) through the `coreml` crate, as the SigLIP2 encoders do; it does not use the ONNX Runtime Core ML execution provider. The two paths share one service and one set of host rules. ONNX Runtime must be 1.28 or later (section 8.2.3). |
| D13 | 2026-10-22. SM2 (model download) runs after SM9. Until SM2, the sidecar reads the model files from a local model directory that the developer fills. SM3, SM7, SM8, and SM9 do not depend on SM2. |
| D14 | 2026-10-22. SM3 is split by function, not by language: SM3.A is the service architecture (Rust and C++), SM3.B is the image embedding cache and the first optimizations, and SM3.C adds the task masks (Subject, Background, Sky, People) to the service. |
| D15 | 2026-10-22. The lab decisions U1 (interaction not worse than Lightroom and Capture One), U3 (no edge post-processing: the edge comes from the model output), U6 (model memory budget 3 GB on Windows, 3.5 GB on macOS; models unload when the user goes back to adjusting), U7 (no interaction that follows the pointer), and U8 (float16 models are acceptable) apply to this plan. The lab MVP UI has known defects; SM9 replans the interaction instead of porting the MVP UI as it is. |

### 1.3 Proposed decisions that need user approval

These items came from the planning discussion. They are not approved. Each item names the phase that
must close it.

| ID | Proposal | Closed by |
| --- | --- | --- |
| P1 | Closed by D12 (2026-10-22): the sidecar runs inference; the C++ process has no ONNX Runtime. | SM1, closed |
| P2 | Store prompts and Mask coverage in normalized ReferenceSpace. | SM5 |
| P3 | Store the R8 coverage only over the bounding rectangle of the Mask content, plus a margin. The maximum axis is `kMaximumRasterMaskAxis` (4096). | SM5 |
| P4 | Closed by U3 (D15): no edge refinement. SM7 stores the model mask after the Feather and Edge adjustment (lab function F8). | SM1, closed |
| P5 | Draw the temporary prompt result in the QSG layer. Show committed coverage through a pipeline coverage preview request. | SM8 and SM9 |
| P6 | Store a geometry fingerprint with each AI Mask. Mark only the Masks whose fingerprint differs from the current Lens and RAW decode geometry. | SM5 |
| P7 | Use one persistent gRPC channel per segmentation session. SM3.A implements the channel (Ping evidence in section 8.1). The second part ("keep only the newest pending decode request while the pointer moves") is dropped: U7 removed decoding during pointer movement, and the tool decodes once per stroke, click, or box. | SM3.A |
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
| Image embedding | The output of the SAM 2.1 image encoder for one analysis render: `image_embed`, `high_res_feats_0`, and `high_res_feats_1`. |
| Prompt | One user input to the SAM 2.1 decoder: a click (one point), a stroke (points sampled along the pointer path), or a box. Each point adds (label 1) or subtracts (label 0). |
| Task mask | A mask of a one-click task: Subject, Background, Sky, or People. |
| Model set | The accepted models of section 8.2 in the platform form: ONNX files on Windows, compiled Core ML models on macOS. |
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
6. The user commits the Candidate Mask. Alcedo Studio applies Feather and Edge, stores the R8
   coverage, and adds one Mask to the selected Color Grade node in one history operation. There is
   no edge refinement stage (D15, U3).
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
| `alcedo-ai-mask-lab` repository | Export the model set to ONNX and Core ML. Check numeric agreement with PyTorch. The export scripts stay in that repository. |
| `rust/puerh_mind` | Load the model set: ONNX Runtime sessions on Windows, native Core ML models on macOS. Keep segmentation sessions. Serve encode, decode, and task mask requests. |
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
| ONNX Runtime | `rust/puerh_mind/Cargo.toml`, `src/service/ort_runtime.rs` | `ort` 2.0.0-rc.12 (ONNX Runtime 1.24) with DirectML and CoreML. The `auto` device lists DirectML then CPU on Windows, and CoreML then CPU on macOS. CUDA is rejected. The C++ process has no ONNX Runtime. | SM3.A raises `ort` to 2.0.0-rc.13 (ONNX Runtime 1.28) because of microsoft/onnxruntime#29071 (section 8.2.3). The segmentation sessions do not use the CLIP execution provider list with its CPU entry. |
| Session lifetime | `src/service/ort_clip.rs`, `coreml_clip.rs`, `encoder_residency.rs` | Encoders load on first use and unload after an idle time. One mutex serializes one encoder. | Add segmentation residency with an explicit unload (SM3.A). |
| Transport | `sidecar_client/client.cpp:57-58` | Plain gRPC over loopback TCP. Each RPC makes a new channel and stub. The client uses the default 4 MB receive limit. | Use a persistent channel for segmentation (SM3). |
| Model catalog | `include/app/model_asset_catalog.hpp`, Rust `model_assets.rs` | The catalog lists semantic (CLIP) profiles only. `BuildAssetUrl` makes `{endpoint}/{repo_id}/resolve/{revision}/{path}`. Assets have size and an optional SHA-256. | Add a segmentation profile kind and a per-profile download location (SM2, after SM9). |
| Download | `app/model_download_service.cpp` | aria2c downloads a profile into a staging directory, verifies it, and promotes it. `StartDownload` accepts only semantic profiles. | Accept segmentation profiles (SM2). |

### 5.5 Project format

`include/edit/history/pipeline_history_format.hpp` states that loaders accept only the current
format identities. The current project metadata version is `0.10.0`, and the Pipeline document JSON
version is 7. A new Mask source kind changes the Pipeline document JSON. SM5 must state the format
change and the result for older projects before implementation.

### 5.6 External model facts

The facts of the first audit (EfficientSAM3 and `facebook/sam3` on Hugging Face) are history; the
SM1 history keeps them (section 8.1). The accepted model set, its files, licenses, and
measurements are in section 8.2.

---

## 6. Target architecture (initial)

This section shows the intended owners. Each later phase confirms or changes its part before
implementation.

```text
Viewport pointer (QML)
  -> EditorMaskCreationAdapter
  -> MaskEditGeometry::MapItemToReference          normalized ReferenceSpace
  -> AI Mask tool controller (app layer)
       ├─ open: sidecar LoadSegmentationModels (SAM 2.1 tiny; SM3.C adds the head and S3OD)
       ├─ start: analysis render (ReferenceSpace, no crop, no rotation)
       │    -> sidecar StartSegmentationSession(image) -> image embedding stays in the sidecar
       ├─ stroke, click, or box: sidecar DecodeSegmentationPrompt(session, prompts, mask_input)
       │    -> Candidate Mask logits -> QSG presentation over the photo
       ├─ Subject, Background, Sky, People (SM3.C): task mask request -> Candidate Mask
       └─ commit: Feather and Edge -> MaskAsset (R8, reference_bounds)
            -> Color Grade owner AddMask operation -> one history operation
  -> close (back to adjusting): sidecar UnloadSegmentationModels (U6)
Sidecar model execution
  Windows: ONNX Runtime 1.28+, DirectML    macOS: native Core ML (.mlmodelc), compute units per model
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
| SM1 | Model selection, export, runtime feasibility, latency, and quality evidence | `alcedo-ai-mask-lab` | none | 900-1500 | complete 2026-10-22 (section 8) |
| SM3.A | Segmentation service architecture: proto, Rust service with the Windows ONNX Runtime and macOS Core ML adapters, SAM 2.1 encode and decode, C++ client with a persistent channel, app session | proto, Rust service, `sidecar_client`, app | SM1 | 2400-2950 estimated, in two parts SM3.A.1 and SM3.A.2; actual about 4,500 | complete 2026-10-22; macOS port check passed with the SM3.B macOS run ([phase plan](ai_segmentation_mask/phase_sm3a_segmentation_service_plan.md)) |
| SM3.B | Image embedding cache and first optimizations | Rust service, app session | SM3.A | 800-1400; actual about 1,800 code lines | complete 2026-10-22 on Windows and macOS ([phase plan](ai_segmentation_mask/phase_sm3b_embedding_cache_plan.md)) |
| SM3.C | Task masks in the service: Subject (S3OD), Background, Sky and People (head) | Rust service, proto, C++ client | SM3.A | 1600-2050 in two parts SM3.C.1 and SM3.C.2; actual about 3,250 code lines | Windows verified 2026-10-23; macOS port check open ([phase plan](ai_segmentation_mask/phase_sm3c_task_masks_plan.md)) |
| SM4 | Analysis render in ReferenceSpace | geometry, scheduler, app service | none | 600-1200 | outline |
| SM5 | AI Mask source model, R8 persistence, history, and project format | `edit/mask`, history, storage | SM1 | over 2000, split before start | outline |
| SM6 | Raster `MaskEvaluate` on CUDA, OpenCL, and Metal | `edit/runtime` | SM5 | 1000-1800 | outline |
| SM7 | Commit: Feather and Edge, asset publication, misalignment notice | app layer, `edit/mask` | SM3.A, SM4, SM5 | 800-1500 | outline |
| SM8 | Coverage preview request | render intent, compiler, three backends | SM6 | 1000-1800 | outline |
| SM9 | AI Mask tool UI and interaction | QML, app controllers | SM3.A-C, SM4, SM7, SM8 | over 2000, split before start | outline |
| SM2 | Segmentation model download from R2 or Hugging Face | model catalog, download service, settings | SM9 | 800-1400 | outline (moved after SM9, D13) |
| SM10 | Product qualification | all | SM1-SM9, SM2 | 300-800 | outline |

Order of work: SM3.A, SM3.B, SM3.C, SM4, SM5, SM6, SM7, SM8, SM9, SM2, SM10. SM4 and SM5 do not
depend on SM3 and can start earlier. Until SM2, the developer fills the local model directory by
hand (D13); the phase plans state the directory and the files.

"Outline" means that the phase has a scope and an estimate only. Each outline phase receives a
detailed phase plan with the alcedo-create-plan skill before it starts. That plan records the user
decisions from section 1.3 that the phase closes.

---

## 8. Phase SM1 — Model selection and runtime feasibility (complete 2026-10-22)

### 8.1 History

SM1 started on 2026-10-19 with EfficientSAM3 and continued with a model re-evaluation. The export
scripts, the runtime measurement program, and the step reports of this work are in the
[`alcedo-ai-mask-lab`](https://github.com/zidage/alcedo-ai-mask-lab) repository. Alcedo Studio does not keep
them. Issues: #448 (SM1), #449 (quality review), #458 (re-evaluation), all closed.

- **EfficientSAM3 proof of concept** (2026-10-19, lab
  [`docs/background/sm1_efficientsam3_record.md`](https://github.com/zidage/alcedo-ai-mask-lab/blob/main/docs/background/sm1_efficientsam3_record.md)):
  the checkpoints export to ONNX and run in the `alcedo_mind` runtime on DirectML and Core ML. The
  full checkpoints contain SAM3 components; only the image encoder trunk is an EfficientSAM3
  student.
- **Quality review failed** (lab
  [`docs/background/model_re_evaluation.md`](https://github.com/zidage/alcedo-ai-mask-lab/blob/main/docs/background/model_re_evaluation.md)): the
  point and box path makes regular grid holes on large regions and fails on close-up portraits.
  The cause is the student image encoder: its native 32 x 32 output is resized to 72 x 72, and the
  decoder is never trained on student features. The ONNX graphs match PyTorch, so the export is
  not the cause.
- **gRPC round trip** (Rust tonic client, 1000 `HealthService.Ping` calls, Windows loopback):
  persistent channel p50 0.20, p95 0.32, maximum 0.60 ms; new channel per call p50 0.52, p95 13.4,
  maximum 26.6 ms. This is the evidence for P7.
- **Re-evaluation steps E1 to E5** (2026-10-19 to 2026-10-20, lab
  [`docs/exploration/`](https://github.com/zidage/alcedo-ai-mask-lab/blob/main/docs/exploration/README.md)): SAM 2.1 removes the grid holes in all
  sizes (E1); BiRefNet is too slow for the Subject task (E2); a short stroke of 3 to 4 points
  selects like a click, and add and subtract strokes work with the previous mask as `mask_input`
  (E3); a live point decoder reaches the intended mask more often than precomputed candidates
  (E4); an edge stage helps only with a second encoder pass (E5). The lab then accepted the model
  set of section 8.2 and decided U3: no edge post-processing.

### 8.2 Accepted model set (2026-10-22)

**Status:** SM1 complete. The re-evaluation continued in the
[`alcedo-ai-mask-lab`](https://github.com/zidage/alcedo-ai-mask-lab) repository (lab commit
`2a33061`). The lab README section "Final selection and MVP architecture", the acceptance record
(`docs/exploration/acceptance_ai_mask_model_set.md`), the MVP report
(`docs/exploration/mvp_ai_mask.md`), and the macOS Core ML report
(`docs/exploration/macos_coreml_verification.md`) are the evidence. This section copies the facts
that later phases need. The lab is the reference for the numbers.

#### 8.2.1 Model set

| Function | Model | Windows file (ONNX Runtime, DirectML) | macOS file (native Core ML) | License |
| --- | --- | --- | --- | --- |
| Stroke, click, box, add and subtract; final masks of Sky and People | SAM 2.1 Hiera tiny image encoder | `point_encoder.onnx`, float32, 109,381,695 bytes, SHA-256 `17a554b420d3c51a109567f784ff3792b354c78dc8c2c09c585403329b412d61` | `point_encoder.mlmodelc`, float32, compute units ALL, 134,260,018 bytes, directory SHA-256 `d3f66d7bcc2a81b2245f22e2ff073b3b1bd180720ea49c091bb0ea401e420285` | Apache-2.0 |
| as above | SAM 2.1 tiny prompt encoder and mask decoder | `point_decoder.onnx`, float32, 21,188,206 bytes, SHA-256 `6f9a0ef2c217a0abccc8b0c01e631c42c6a78a2038a7944cc7fcb4f536439278` | `point_decoder.mlmodelc`, float32, compute units CPU_ONLY, 1 to 64 points, 21,098,274 bytes, directory SHA-256 `f2efed0420fba10b641afa384a58a872cc878231472711e9d86219bf4440fd70` | Apache-2.0 |
| Sky and People localization | People and sky head, P1 run 2, epoch 15 | `people_sky_head.onnx`, 1,119,131 bytes, SHA-256 `e02f2a1eaace94e0efe2b3c8db0a0391be210b477a95e49063e80758f12abf35`; thresholds `people_sky_head.json`, SHA-256 `43981e9ed1336ee97bcd3b7a7f07318523a56d43a24631c981d29d27c4fb2486` | `people_sky_head.mlmodelc`, compute units ALL, 1,138,418 bytes, directory SHA-256 `ccb5c48d19d8785ca9fc485d01b3b2ef94989de480fb2eba6c9471b0728540c9` | Own model (COCO 2017, Open Images V7) |
| Subject; Background = 1 - Subject | S3OD `s3od-synth` | `subject_masks.onnx`, float16 with the encoder layer 0 attention in float32, attention in 8 chained query chunks, 219,301,899 bytes, SHA-256 `0f070b8641676d68faaafc0b70816be1458fcc976d8d6d119e67038527cbc288` | `s3od.mlmodelc` (`float16_embeddings_attention_float32`), compute units ALL, 271,166,829 bytes, directory SHA-256 `2c6a24e61ca5dc062ed80a5163974e0e58500fd94fc02b0e078eba7205c0424d` | MIT; DINOv3 License for the encoder |

The directory SHA-256 of a `.mlmodelc` is the lab rule: SHA-256 over the sorted relative paths and
the file hashes (`convert_models.directory_sha256`). SM2 decides the published packaging of the
Core ML directories.

Model input and output names (the Core ML models use the ONNX input names; the decoder output names
differ):

| Model | Inputs | Outputs |
| --- | --- | --- |
| Encoder | `pixel_values` [1, 3, 1024, 1024]: the photo stretched to 1024 x 1024 (bilinear with antialias, no padding), ImageNet mean and standard deviation | `image_embed` [1, 256, 64, 64], `high_res_feats_0` [1, 32, 256, 256], `high_res_feats_1` [1, 64, 128, 128] |
| Decoder | the three encoder outputs, `point_coords` [1, N, 2] (normalized photo coordinates x 1024 on each axis), `point_labels` [1, N] (1 add, 0 subtract, 2 and 3 box corners, box first), `mask_input` [1, 1, 256, 256], `has_mask_input` [1] | ONNX: `mask_logits` [1, 4, 256, 256], `iou_scores` [1, 4], `object_score_logits` [1, 1]. Core ML: `masks`, `iou_predictions`, `object_score_logits` with the same shapes |

#### 8.2.2 Decisions for the items that SM1 prepared

| Item | Decision |
| --- | --- |
| Checkpoint variant | SAM 2.1 Hiera tiny (E1, E3, E5; lab README section 4). EfficientSAM3 is not used. |
| Text prompt support in 0.3.4 | None (D2). No model of the set takes text. |
| P1: sidecar or C++ process | Sidecar (D12). |
| Core ML use on macOS | Native Core ML models, not the ONNX Runtime Core ML execution provider (lab U15). The provider runs S3OD on the CPU (7.9 s, 10.3 GB) and stops the process with the decoder in the MLProgram format. |
| Acceptable CPU operator placement | Windows DirectML runs the set with the MVP session settings (one execution provider, no CPU entry). macOS runs the decoder on CPU_ONLY by decision (about 30 ms per call, no cost for a change of the point count; U15). |
| P4: edge refinement method | None (U3). The edge comes from the model output. |
| Model input size | SAM 2.1: 1024 x 1024 stretch. S3OD: letterbox to a 1024 long edge. The photo enters at a long edge of at most 4096 px. |
| Published file list | Section 8.2.1. |

#### 8.2.3 Runtime facts that later phases must keep

- ONNX Runtime 1.24.4 to 1.27.0 fold the two `If` nodes of the SAM 2.1 tiny encoder into freed
  memory (microsoft/onnxruntime#29071, fixed in 1.28.0). On the Mac the encoder output was wrong in
  every process; on Windows the result depends on the memory reuse. Alcedo Studio used ONNX Runtime
  1.24 (`ort` 2.0.0-rc.12, `api-24`). SM3.A raises it to 1.28 (`ort` 2.0.0-rc.13, `api-28`).
  `ort` 2.0.0-rc.14 and later need Rust 1.92.
- MVP session settings on ONNX Runtime: one execution provider per session and no other provider
  after an error; 4 intra-op CPU threads, 1 inter-op thread; CPU memory arena off; DirectML with
  sequential execution and memory patterns off.
- On DirectML, a decoder session runs only the point count of its first run at about 10.5 ms; other
  counts take 22 to 34 ms. Padding to a fixed point count breaks the masks. One session per point
  count gives 12 ms at about 90 MB each (E3). SM3.B implements one session for each of the
  counts 1, 2, and 4 and one general session, each run at load (SM3.B plan, decision B3).
- Core ML prepares the decoder again for each new point count on the GPU (about 52 ms); on the CPU
  the decoder takes about 30 ms per call. Keep all Core ML calls of one model on one thread.
- Decoder mask selection: one point without a box (a click) returns the token 1 to 3 with the best
  `iou_scores`, and the three tokens sorted by area are the levels of a level change without a model
  run. A box or several points use token 0 when its stability (area of logits > 0.05 / area of
  logits > -0.05) is at least 0.98, else the token 1 to 3 with the best score. Logits are clipped to
  +-32. A mask at the photo size is the bilinear upsampling of the 256 x 256 logits, threshold 0.
- Each stroke, click, or box decodes all points and the box of the selection in one decoder call,
  with the previous selected logits as `mask_input` (`has_mask_input` = 1). A stroke is 4 points at
  equal spacing along the pointer path; a click is 1 point.

#### 8.2.4 Cost of the accepted set

All models loaded in one process, warm p50 (lab README, section "Cost of the MVP pipeline"):

| Stage | Windows, RTX 3080 Laptop, DirectML | macOS, Apple M4, native Core ML |
| --- | ---: | ---: |
| Session load: encoder, decoder, head, S3OD | 2,623, 249, 112, 910 ms | 1,197, 69, 9, 43 ms |
| Encoder (once per photo) | 70.5 ms | 191 ms |
| Head (once per photo) | 2.0 ms | 2.1 ms |
| Decoder: box; 10 points with `mask_input` | 23.2; 22.2 ms | 33.5; 34.1 ms |
| S3OD (once per photo) | 191.1 ms | 531 ms |
| Memory with all models loaded | 2,666 MB dedicated GPU memory | 3,190 MB peak footprint of the Python process |

#### 8.2.5 Lab MVP

The lab contains a PySide6 MVP of the AI Mask tool (`scripts/poc/ai_mask_mvp/`): the pipeline
`mask_pipeline.py` (no UI) and the UI `ai_mask_app.py`. The pipeline is the reference for the host
rules of SM3.A and SM3.C: the port is correct when it gives the MVP results on the 14 quality photos
(lab README, section "Port checks"). The MVP UI is a reference for SM9 only (D15).

#### 8.2.6 Open items and their owner phases

| Item | Owner phase |
| --- | --- |
| The user check of the interaction by eye on Windows and macOS (U1) | SM9 |
| Subject "not found" rule (F1); People list threshold and order (F4a) | SM3.C; closed by C1 to C3 of the SM3.C plan |
| Decoder point limit of a long selection (the Core ML decoder takes 1 to 64 points; each stroke adds 4) | SM3.A rejects more than 64 points; SM9 decides the tool behavior at the limit |
| Combination across mask types (F7) and recipe storage (F9), lab decision O1 | SM5 and SM9, after a user decision |
| Training data rule U13 and the S3OD float32 tolerance (lab acceptance record, section 6) | User decision before SM2 publishes files |
| Model unload when the user goes back to adjusting (U6) | SM3.A provides the operation; SM9 calls it |

---

## 9. Outline phases

Each outline phase lists its scope and its open decisions. A detailed phase plan replaces the outline
before the phase starts.

### 9.1 SM2 — Segmentation model download (after SM9)

SM2 runs after SM9 (D13). Until then, the sidecar reads the model files from the local model
directory of SM3.A.

Scope:

- Add a segmentation profile kind to `model_asset_catalog` and to Rust `model_assets.rs`. Keep the two
  catalogs in agreement.
- Add a download location per profile, so that semantic profiles keep the current endpoint setting
  and the segmentation profile reads from R2 or a Hugging Face repository.
- Publish the Windows files (ONNX) and the macOS files (compiled Core ML directories) of section
  8.2.1. Decide the packaging of the Core ML directories (for example one zip per model, as the
  SigLIP2 Core ML profiles use).
- Pin the SHA-256 of every segmentation file.
- Download the license files with the model files (Apache-2.0, MIT, DINOv3 License).
- Install into the local model directory that SM3.A reads, so that no sidecar change is necessary.
- Replace the SM9 "model not installed" error state with the download panel of section 3.1 and in
  the settings.

Open decisions: download location (R2, Hugging Face, or both), settings page placement, the
training data rule U13 and the S3OD float32 tolerance (section 8.2.6).

### 9.2 SM3 — Sidecar segmentation service and C++ client

SM3 is split by function (D14). Each part changes Rust and C++ code as its function needs.

#### 9.2.1 SM3.A — Service architecture

Detailed plan: [SM3.A Segmentation service architecture](ai_segmentation_mask/phase_sm3a_segmentation_service_plan.md). Issues: SM3.A.1 #460,
SM3.A.2 #461.

Scope: `segmentation.proto`; the Rust segmentation service with one backend interface and two
adapters (ONNX Runtime with DirectML on Windows, native Core ML on macOS); explicit model load and
unload; segmentation sessions with one image embedding each; SAM 2.1 encode and decode with the host
rules of section 8.2.3; the C++ segmentation client with a persistent channel and a receive limit
that fits the largest message; the app-layer segmentation session; the ONNX Runtime 1.28 upgrade.

#### 9.2.2 SM3.B — Image embedding cache and first optimizations

Detailed plan: [SM3.B Image embedding cache and first optimizations](ai_segmentation_mask/phase_sm3b_embedding_cache_plan.md).
The user closed the open decisions on 2026-10-22: the cache key is an opaque host image key with
a key-only start (B1; SM4 defines its content), the cache keeps 4 embeddings and unload clears it
(B2), and DirectML loads decoder sessions for the point counts 1, 2, and 4 and one general
session (B3).

Scope:

- Keep the image embedding of recent analysis renders, so that a new session on the same image
  content does not run the encoder again. Define the cache key from the analysis render identity
  of SM4 and the model set identity. Define the memory limit against U6.
- Keep the encoder outputs in the native form of the backend (ONNX Runtime values, Core ML
  multi-arrays) between decoder calls, so that a decoder call does not copy 16 MB of features.
- DirectML: one decoder session per point count (section 8.2.3), with a limit on the number of
  sessions and their memory.
- Measure each change against the SM3.A latency record on Windows DirectML and macOS Core ML.

Open decisions: none (closed by B1 to B3 of the SM3.B plan).

#### 9.2.3 SM3.C — Task masks in the service

Detailed plan: [SM3.C Task masks in the segmentation service](ai_segmentation_mask/phase_sm3c_task_masks_plan.md).
The user closed the open decisions on 2026-10-23: Subject is "not found" only when no alpha value
reaches 0.5 (C1, F1); the People list uses the list threshold of the head and keeps the best box
when the person presence passes and no box reaches the threshold (C2, F4a); the list is sorted
from left to right (C3, F4a); the sidecar makes the Sky union and the union of the selected people
(C4). The union is the OR of the member masks at the session image size, returned as an R8 mask
(C5).

Scope:

- Load the people and sky head and S3OD with the SAM 2.1 models (section 8.2.1), on both
  platforms.
- Run the head once per session on `image_embed`. Run S3OD once per session at the first Subject
  or Background request.
- Add the task mask requests: Subject (the S3OD mask with the largest IoU score, unpadded),
  Background (1 - Subject), Sky (presence threshold, 8-connected regions, 3 points per region, one
  decoder call per region, union), and People (presence threshold, person boxes, one box prompt per
  person, People list).
- Decide the Subject "not found" rule (F1) and the People list threshold and order (F4a).
- Verify the port against the lab MVP results on the 14 quality photos (section 8.2.5).

Open decisions: none (closed by C1 to C4 of the SM3.C plan).

### 9.3 SM4 — Analysis render in ReferenceSpace

Scope:

- Add a render mode that applies no crop and no rotation, or prove that an existing mode gives the
  same frame. `RotatedUncroppedSource` keeps the rotation (section 5.3).
- Render the whole frame at the model input size, and a ReferenceSpace rectangle at up to 4096 on
  the long edge for edge refinement.
- Read the result back to the host with `require_host_output`.

Open decisions: which pipeline stages the analysis render applies (for example Develop and the
default display transform without Color Grade nodes), and the content of the image key of the
SM3.B embedding cache (the key names the analysis render content and the model set; SM3.B plan,
decision B1).

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

- Make the coverage from the Candidate Mask: bilinear upsampling of the 256 x 256 logits
  (threshold 0) or of the S3OD alpha, then Feather and Edge (lab function F8). There is no edge
  refinement stage (U3).
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

- The AI Mask tool: stroke, click, box, add and subtract, level change, the task masks of SM3.C,
  Candidate Mask presentation, commit, and cancel.
- Model load when the tool opens and unload when the user goes back to adjusting (U6).
- The busy state and the error states of section 3.3. Until SM2, a missing model set shows the real
  "model not installed" error with the expected local model directory; SM2 adds the download panel.
- The Mask thumbnail from the R8 asset.
- Use the lab MVP UI as a reference only (D15). Record each difference from it with its reason.

Open decisions: the exact interaction set of D3, and the tool behavior at the 64-point decoder
limit. Split this phase before it starts. The expected diff is more than 2000 lines.

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
| ONNX Runtime before 1.28 computes the SAM 2.1 tiny encoder wrongly. | A build with `ort` older than 2.0.0-rc.13, or a port check failure on Windows. | Keep ONNX Runtime 1.28 or later. Do not add a workaround in the graph or the session settings. |
| The Core ML adapter cannot be built or run while the Mac is not available. | SM3.A completion record. | Report the macOS build and run as not done. Run them on the Mac before SM3.B is complete. |
| DirectML runs operators on the CPU provider. | SM1 operator placement report; SM3.A sessions have no CPU entry, so a failed placement is a load error. | Report the error. The user decides. Do not change the device without a user decision. |
| Whole-image input is too coarse for small subjects. | SM1 cases 5 and 6. | Report the IoU. D7 stays. Propose a refinement method in SM7. |
| The edge quality is low at full resolution. | SM9 user check by eye. | Report it. U3 rejects edge post-processing; a change needs a new user decision. |
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
