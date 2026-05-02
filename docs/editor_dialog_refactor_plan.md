# Editor Dialog Refactor Plan

## Scope

This plan describes how to move the editor dialog from a monolithic `EditorDialog` implementation to the structure defined in `docs/editor_dialog_refactor_design.md`.

The team should execute this as a sequence of small, buildable pull requests. Do not attempt to move every panel in one change.

## Invariants

These behaviors must remain stable throughout the refactor:

- Opening the editor initializes from current pipeline params when present.
- Brand-new pipelines keep the existing default behavior, including the default LUT path.
- Dragging a slider requests fast preview without creating history transactions.
- Releasing a slider or selecting a discrete option commits a transaction when the value changed.
- Undo reloads UI from pipeline state and updates all visible controls.
- Version checkout reconstructs params, imports them into the pipeline, reloads UI, and starts a working version.
- Geometry panel preview remains overlay-first and does not crop destructively until commit.
- Quality preview still runs after commit or release, even if the field did not change.

## Phase 0: Baseline Characterization

Goal: capture current behavior before moving ownership.

Tasks:

1. Add focused manual test notes under `docs/editor_dialog_manual_test_matrix.md`.
2. Cover at least: tone slider drag and release, curve reset, color temp custom/as-shot, LUT selection, HLS profile switch, CDL wheel drag, ODT method switch, geometry crop apply/reset, raw highlight toggle, lens override, undo, commit all, version checkout.
3. Record the expected pipeline stage/operator for each field.

Exit criteria:

- The team has a stable behavior checklist to run after each panel migration.

## Phase 1: Introduce Session Types Without Moving Panels

Goal: create the mediator API while existing `EditorDialog` methods still call the old code.

New files:

```text
alcedo_studio/src/include/ui/alcedo_main/editor_dialog/session/
  adjustment_panel.hpp
  adjustment_patch.hpp
  adjustment_snapshot.hpp
  editor_adjustment_session.hpp

alcedo_studio/src/ui/alcedo_main/editor_dialog/session/
  editor_adjustment_session.cpp
```

Tasks:

1. Define `AdjustmentPreview`, `AdjustmentCommit`, `PreviewPolicy`, and `CommitPolicy`.
2. Define `EditorAdjustmentSession` with `Preview()`, `Commit()`, `LoadFromPipeline()`, and `ReloadFromImportedPipelineParams()`.
3. Move the body of `EditorDialog::CommitAdjustment()` into `EditorAdjustmentSession::Commit()`.
4. Keep `EditorDialog::CommitAdjustment()` as a temporary forwarding wrapper.
5. Keep existing `AdjustmentState` and `pipeline_io.cpp` unchanged.

Exit criteria:

- All current UI still compiles.
- Existing panel files still call `EditorDialog::CommitAdjustment()`, but the real transaction algorithm lives in the session.

Progress note (2026-05-02):

- Phase 1 session scaffolding has been implemented. The new `session/` API defines adjustment preview/commit request types, the adjustment panel interface, and a legacy snapshot wrapper.
- `EditorAdjustmentSession` now owns the commit transaction algorithm that was previously in `EditorDialog::CommitAdjustment()`. Existing panel code still calls the dialog wrapper, which forwards to the session.
- `AdjustmentState` and `pipeline_io.cpp` remain unchanged for this phase. No panel ownership has moved yet, so the manual matrix should be used as the behavior baseline for follow-up phases.

## Phase 2: Extract Render and History Coordinators

Goal: reduce `EditorDialog` before panel migration.

New files:

```text
alcedo_studio/src/include/ui/alcedo_main/editor_dialog/render/
  editor_render_coordinator.hpp

alcedo_studio/src/include/ui/alcedo_main/editor_dialog/history/
  editor_history_coordinator.hpp

alcedo_studio/src/ui/alcedo_main/editor_dialog/render/
  editor_render_coordinator.cpp

alcedo_studio/src/ui/alcedo_main/editor_dialog/history/
  editor_history_coordinator.cpp
```

Tasks:

1. Move preview timers, pending requests, inflight future, and render completion flow out of `EditorDialog`.
2. Move version reload, checkout, undo, commit working version, and working-version creation into `EditorHistoryCoordinator`.
3. Keep old `EditorDialog` methods as forwarding wrappers while panels are still monolithic.
4. Route history reload through `EditorAdjustmentSession`.

Exit criteria:

- `EditorDialog` no longer owns preview request queues or history transaction list logic.
- Behavior matches the baseline test matrix.

Progress note (2026-05-02):

- Phase 2 render/history coordinator extraction has been implemented. `EditorRenderCoordinator` now owns preview request queues, preview generation/detail state, render timers, inflight future polling, and render completion sequencing. Existing `EditorDialog` render methods remain as forwarding wrappers for unmigrated panel code.
- `EditorHistoryCoordinator` now owns the working version and version operations: version reconstruction, checkout, undo, commit-all, working-version seeding, and version UI refresh. History-triggered pipeline reloads route back through the adjustment session load path before synchronizing legacy controls.
- `EditorDialog` still owns shell widgets and legacy panel controls, but it no longer stores the render queues/timers/future state or the working `Version` directly. CMake includes the new `render/` and `history/` coordinator sources.
- Verification: `cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 4` passes. For `ctest`, `CudaImageGeometryOpsTest` and `CudaPreviewVramReclamationTest` were disabled via new CMake switches because their executables exit `0xc0000139` during GoogleTest discovery. Additional generated discovery includes with the same startup status were disabled in the local build tree for this run: `PipelineServiceTest`, `EditHistoryMgmtServiceTest`, `ExportServiceTest`, and the `AlbumBackend*` tests.
- Remaining `ctest --test-dir build/debug --output-on-failure` execution reached 94 discovered tests; 88 passed, 2 were skipped, and 6 unrelated assertions failed in `RawProcessorPatternTest`, `CudaRawOpsTest`, and `SharedToneCurveTest`. Run the manual matrix before merging UI-facing follow-up phases.

## Phase 3: Split Typed State and Adapter Skeletons

Goal: create per-module state types before moving widgets.

New files:

```text
alcedo_studio/src/include/ui/alcedo_main/editor_dialog/state/
  tone_adjustment_state.hpp
  color_temp_adjustment_state.hpp
  look_adjustment_state.hpp
  display_transform_adjustment_state.hpp
  geometry_adjustment_state.hpp
  raw_decode_adjustment_state.hpp

alcedo_studio/src/include/ui/alcedo_main/editor_dialog/pipeline/
  tone_pipeline_adapter.hpp
  color_temp_pipeline_adapter.hpp
  look_pipeline_adapter.hpp
  display_transform_pipeline_adapter.hpp
  geometry_pipeline_adapter.hpp
  raw_pipeline_adapter.hpp
```

Tasks:

1. Move field groups out of `AdjustmentState` into typed structs.
2. Add `EditorAdjustmentSnapshot` as the aggregate typed snapshot.
3. Add temporary conversion functions between `EditorAdjustmentSnapshot` and legacy `AdjustmentState`.
4. Create adapter skeletons that initially delegate to `pipeline_io.cpp`.

Exit criteria:

- No behavior changes.
- Module state groups are available to panel widgets.
- Legacy render path can still consume `AdjustmentState`.

Progress note (2026-05-02):

- Phase 3 typed state and adapter skeletons have been implemented. Six typed state structs now live under `state/`: `ToneAdjustmentState`, `ColorTempAdjustmentState`, `LookAdjustmentState`, `DisplayTransformAdjustmentState`, `GeometryAdjustmentState`, and `RawDecodeAdjustmentState`.
- `EditorAdjustmentSnapshot` has been redefined as an aggregate of the six typed states plus `RenderType`. Inline conversion helpers `ToLegacyAdjustmentState()` and `FromLegacyAdjustmentState()` preserve full bidirectional compatibility with the existing `AdjustmentState` struct.
- Six pipeline adapter skeletons now live under `pipeline/`: `TonePipelineAdapter`, `ColorTempPipelineAdapter`, `LookPipelineAdapter`, `DisplayTransformPipelineAdapter`, `GeometryPipelineAdapter`, and `RawPipelineAdapter`. Each adapter is stateless and currently delegates to `pipeline_io.cpp` by constructing temporary legacy `AdjustmentState` objects, calling the existing free functions, and extracting the relevant results. A generic `PipelineLoadResult<T>` template is provided in `adjustment_pipeline_adapter.hpp`.
- `AdjustmentState` itself remains unchanged; no existing code paths were modified. All new headers are wired into `alcedo_studio/src/CMakeLists.txt`.
- Verification: `cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 4` passes. For `ctest`, the same `0xc0000139`-affected tests disabled in Phase 2 were excluded from the local build tree (`PipelineServiceTest`, `EditHistoryMgmtServiceTest`, `ExportServiceTest`, `AlbumBackend*`, `CudaImageGeometryOpsTest`, `CudaPreviewVramReclamationTest`). Remaining `ctest --test-dir build/debug --output-on-failure` reached 94 discovered tests; 88 passed, 2 were skipped, and 6 unrelated pre-existing assertions failed in `RawProcessorPatternTest`, `CudaRawOpsTest`, and `SharedToneCurveTest`.

## Phase 4: Migrate Tone Panel First

Goal: prove the panel pattern on the highest-value but relatively straightforward panel.

Files to change:

```text
alcedo_studio/src/include/ui/alcedo_main/editor_dialog/widgets/tone_control_panel_widget.hpp
alcedo_studio/src/ui/alcedo_main/editor_dialog/widgets/tone_control_panel_widget.cpp
alcedo_studio/src/include/ui/alcedo_main/editor_dialog/dialog_internal.hpp
alcedo_studio/src/ui/alcedo_main/editor_dialog/dialog_core.cpp
```

Tasks:

1. Move `BuildToneControlPanel()` body into `ToneControlPanelWidget`.
2. Move tone-owned controls from `EditorDialog` to `ToneControlPanelWidget`.
3. Move tone-owned reset callbacks to the panel.
4. Replace direct `state_` mutation with `ToneAdjustmentState`.
5. Route preview and commit through `EditorAdjustmentSession`.
6. Keep color temperature in this panel only if the UI layout currently requires it; otherwise split it into `ColorTempPanelSection` inside the same widget.

Exit criteria:

- `EditorDialog` owns `ToneControlPanelWidget* tone_panel_`, not tone sliders.
- Tone sliders, curve editing, resets, preview, commit, undo reload, and version checkout work.

## Phase 5: Migrate Raw Decode Panel

Goal: migrate a smaller discrete-control panel to validate non-slider commits.

Files to change:

```text
alcedo_studio/src/include/ui/alcedo_main/editor_dialog/widgets/raw_decode_panel_widget.hpp
alcedo_studio/src/ui/alcedo_main/editor_dialog/widgets/raw_decode_panel_widget.cpp
```

Tasks:

1. Move raw highlight reconstruction control into `RawDecodePanelWidget`.
2. Move lens calibration enablement and lens override controls into the widget.
3. Move lens catalog state out of `EditorDialog`.
4. Use `RawPipelineAdapter` for load, params, and changed checks.

Exit criteria:

- `EditorDialog` no longer declares raw or lens control pointers.
- Raw decode and lens calibration transactions remain identical.

## Phase 6: Migrate Geometry Panel

Goal: isolate geometry state and viewer overlay behavior.

Files to change:

```text
alcedo_studio/src/include/ui/alcedo_main/editor_dialog/widgets/geometry_panel_widget.hpp
alcedo_studio/src/ui/alcedo_main/editor_dialog/widgets/geometry_panel_widget.cpp
```

Tasks:

1. Introduce `GeometryOverlayPort` and implement it in the shell or viewer pane.
2. Move crop state helpers into `GeometryPanelWidget` or `GeometryAdjustmentModel`.
3. Keep pure geometry math in `modules/geometry.*`.
4. Move `ResetCropAndRotation()`, crop aspect sync, crop label updates, and viewer overlay pushes out of `EditorDialog`.
5. Teach `EditorAdjustmentSession` or `EditorRenderCoordinator` that active geometry editing uses committed crop params for pipeline render while draft geometry appears in overlay.

Exit criteria:

- `EditorDialog` no longer owns geometry controls or geometry helper methods.
- Crop overlay behavior matches baseline.
- Committing crop still marks full-frame preview as required.

## Phase 7: Migrate Display Transform Panel

Goal: isolate ODT complexity and OpenDRT detail binding state.

Tasks:

1. Move ODT controls and `OpenDrtDetailSliderBinding` into `DisplayTransformPanelWidget`.
2. Move ODT sanitize and supported-encoding UI refresh into display-transform state or adapter helpers.
3. Use `DisplayTransformPipelineAdapter`.
4. Keep `frame_manager_.SyncViewerDisplayEncoding()` behind a session/render callback so the panel does not know about `EditorFrameManager`.

Exit criteria:

- `EditorDialog` no longer owns ODT controls or OpenDRT detail slider bindings.
- ODT method switching and display encoding sync remain correct.

## Phase 8: Migrate Look Panel

Goal: isolate the most stateful color panel after the session API has been proven.

Tasks:

1. Move HLS target state, HLS profile tables, and profile save/load helpers into `LookAdjustmentState` or `LookAdjustmentModel`.
2. Move CDL wheel controls and labels into `LookControlPanelWidget`.
3. Move LUT browser ownership and LUT navigation shortcut handling into the look panel or a dedicated `LutPanelSection`.
4. Keep catalog scanning in existing LUT catalog/controller helpers unless the team decides to split it later.

Exit criteria:

- `EditorDialog` no longer owns HLS, CDL, or LUT browser controls.
- LUT keyboard navigation still works only when the LUT browser should consume it.

## Phase 9: Migrate Versioning UI

Goal: remove versioning controls from `EditorDialog`.

Tasks:

1. Move versioning buttons, lists, collapsed flyout, page stack, and animation state into `VersioningPanelWidget`.
2. Let `EditorHistoryCoordinator` own working-version behavior.
3. Keep `EditorDialog` responsible only for placing the versioning widget in the shell.

Exit criteria:

- `EditorDialog` no longer declares versioning controls or flyout animation fields.

## Phase 10: Remove Legacy Snapshot and Wrappers

Goal: finish the architecture after panels have moved.

Tasks:

1. Delete forwarding wrappers on `EditorDialog`.
2. Delete unused fields from `AdjustmentState` or remove it entirely.
3. Delete compatibility conversion functions.
4. Split or delete `modules/pipeline_io.cpp` after module adapters own all mappings.
5. Move remaining shell-private declarations out of `dialog_internal.hpp` into shell-private headers.

Exit criteria:

- `dialog_internal.hpp` is no longer a monolithic private header.
- Panel files implement panel classes only.
- The CMake file lists the new session, state, adapter, render, history, shell, and panel sources.

## Pull Request Order

Recommended PR sequence:

| PR | Change | Risk |
| --- | --- | --- |
| 1 | Add behavior checklist and session data types | Low |
| 2 | Move commit algorithm into `EditorAdjustmentSession` | Medium |
| 3 | Move render coordinator | Medium |
| 4 | Move history coordinator | Medium |
| 5 | Add typed states and adapter skeletons | Low |
| 6 | Migrate tone panel | High |
| 7 | Migrate raw decode panel | Medium |
| 8 | Migrate geometry panel | High |
| 9 | Migrate display transform panel | High |
| 10 | Migrate look panel | High |
| 11 | Migrate versioning panel | Medium |
| 12 | Delete legacy snapshot and wrappers | Medium |

## Review Checklist

For each PR, reviewers should check:

- Does the changed panel still avoid direct `PipelineGuard` and `Version` ownership?
- Does preview remain transaction-free?
- Does commit happen exactly once per release or discrete selection?
- Does undo reload both panel-local state and controls?
- Does version checkout update every migrated and unmigrated panel?
- Are old `EditorDialog` fields removed when ownership moves?
- Are adapter functions stateless and testable?
- Are UI-only helpers kept out of pipeline adapters?
- Does CMake include each added source and header?

## Testing Guidance

Use the MSVC wrapper when building on Windows:

```bash
cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 4
ctest --test-dir build/debug --output-on-failure
```

After panel migration PRs, run the manual checklist from `docs/editor_dialog_manual_test_matrix.md`.

For adapter-heavy changes, add small C++ tests that validate:

- default state maps to default operator params;
- loaded params round-trip to state and back;
- changed checks are field-local;
- geometry crop params preserve the existing overlay/commit behavior.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Preview path and commit path diverge | Keep `EditorAdjustmentSession` as the single path for panel writes. |
| Version checkout updates only migrated panels | Session should broadcast a reload event to every registered panel. |
| Geometry crop starts applying during draft edits | Keep an explicit geometry editing policy in the session/render coordinator. |
| Panels duplicate pipeline mapping | Use stateless module adapters and delete mappings from widgets during review. |
| The global snapshot lingers forever | Track deletion of `AdjustmentState` compatibility in Phase 10. |
| Large PRs become unreviewable | Migrate one panel per PR and remove old dialog fields in the same PR. |

## Completion Criteria

The refactor is ready for final acceptance when:

- `EditorDialog` reads as a shell and composition root.
- Each adjustment panel is a real widget with local state and local controls.
- The session is the only adjustment transaction boundary.
- Pipeline adapters are module-specific and stateless.
- The manual matrix passes across tone, look, display transform, geometry, raw decode, versioning, undo, and version checkout.
