# AI Sidecar Runtime Residency Plan

Date: 2026-10-17

Status: planned. No phase has implementation evidence.

Source revision of the audit: `main` at `ada6f002c`.

Primary owner: Alcedo Studio AI runtime (`AiSidecarRuntimeService` and the `alcedo_mind` Rust
sidecar).

Issue: [#424](https://github.com/zidage/AlcedoStudio/issues/424). Related: [#423](https://github.com/zidage/AlcedoStudio/issues/423) (global search redesign).

Affected areas:

- `app/ai_sidecar_runtime_service` (process lifetime)
- `app/project_service` (`ProjectSemanticSearchProvider`, runtime creation)
- `ui/alcedo_main/album_backend/image_analysis_environment`, `image_analysis_controller`,
  `semantic_generation_controller` (runtime callers)
- `ui/alcedo_main/album_backend/search_controller` (prewarm)
- `qml/SemanticGenerationSettingsPanel.qml` (setting)
- `rust/puerh_mind/src/service/ort_clip.rs`, `coreml_clip.rs`, `inference.rs`,
  `embedding.rs`, `server/semantic.rs`, `config.rs`, `proto/semantic.proto`

Related plans:

- [AI Sidecar Backend Integration Plan](ai_sidecar_backend_plan.md). This plan extends its
  runtime service (Phase 6e). It does not change the client modules.
- [Semantic Generation and Search Integration Plan](semantic_generation_search_plan.md). This
  plan replaces its runtime lifecycle rule ("start for an explicit user action, stop after the
  last lease") with the idle stop rule of Section 2.1.
- [AI Sidecar Frontend Plan](../ui/ai_sidecar_frontend_plan.md). Its note that the host calls
  `Stop()` before `StartAndWait()` to reload provider configs does not match the current source.
  Phase SR1 adds the provider config restart rule.
- [Global Search Redesign Plan](../ui/global_search_redesign_plan.md). That plan applies a
  natural-language search on Enter. This plan makes that search fast after the first one, and
  adds the prewarm when the palette opens.

## 1. Decision record

### 1.1 Product goal

A natural-language search, a semantic generation run, and an image analysis run must not load
the model again for each operation. The sidecar process and the CLIP encoders stay loaded for a
configurable idle time after the last use. Memory use stays limited: only the encoder that an
operation needs is loaded, and each encoder unloads after its own idle time.

### 1.2 Approved decisions

1. **Idle time, not a k-LRU.** A project has one active CLIP model, and remote providers have no
   local weights. An LRU over models has one entry. The residency rule is an idle time for the
   process and an idle time for each encoder.
2. **Two encoders, two rules.** The text encoder serves search and label prototypes. It is small
   and latency-sensitive. It loads on the CPU and has the long idle time. The image encoder
   serves embedding generation. It is large and runs in batches. It loads on the configured
   device and unloads after 60 seconds of idle time.
3. **Lazy load.** The sidecar loads no encoder at start. It loads an encoder on the first request
   that needs it. Image analysis with remote providers loads no encoder.
4. **User setting.** The setting "Keep AI model loaded" has the values: unload after use,
   1 minute, 5 minutes (default), 15 minutes, and while the project is open. It sets the process
   idle time and the text encoder idle time.
5. **Prewarm.** When the search palette opens and natural-language search is on, the host starts
   the sidecar and loads the text encoder in the background.
6. **Text encoder device.** The text encoder uses the CPU on all platforms. Reason: one query is
   short, and the CPU keeps the text encoder out of the GPU memory that the render pipeline
   uses. If the CPU session fails to load, the request fails with that error. The sidecar does
   not try another device.
7. **This is a separate plan from the global search redesign.** Search works without this plan.

### 1.3 Rejected options

- An LRU over models or over projects.
- A sidecar that runs for the whole application lifetime with no idle stop.
- Loading the text encoder on DirectML or Core ML by default.
- A retry on another device after an encoder load failure.

## 2. Product design specification

### 2.1 Process idle stop

- `AiSidecarRuntimeService` counts leases, as today.
- When the last lease is released, the service starts the idle stop timer with the configured
  delay.
  - Delay 0: the service stops the process at once (current behavior).
  - Delay greater than 0: the service stops the process when the timer expires and no lease
    exists.
  - Delay "while the project is open": no timer. The process stops at project close.
- A new lease stops the timer.
- The idle stop does not block the GUI thread. It sends the terminate request, and it finishes
  the state change when the process exits. It kills the process when the graceful stop time
  passes.
- Project close stops the process at once, as today (`StopForProjectClose`).

### 2.2 Restart rules

These rules exist because the process now stays alive between operations.

- **Model identity change.** A start request with a different `model_root`, `model_id`, or
  `revision` than the running process, and no lease other than the caller's own, restarts the
  process with the new options. With other leases, the process is reused, as today, and the
  caller receives the current model mismatch error.
- **Provider config change.** When the provider configs change (`SidecarConfigsDirty`), the next
  start request with no other lease restarts the process. With other leases, the restart happens
  at the next start after the lease count reaches 0.
- **Text encoder device or idle time change.** These are process arguments. They apply at the
  next process start.

### 2.3 Encoder residency in the sidecar

| Encoder | Device | Load | Idle unload | Process argument |
| --- | --- | --- | --- | --- |
| Text | `cpu` | first text request | setting value (Section 1.2, item 4); never for "while the project is open"; at once after each request for "unload after use" | `--text-device`, `--text-idle-unload-seconds` |
| Image | `--device` (default `auto`) | first image request | 60 s | `--vision-device`, `--vision-idle-unload-seconds` |
| Multimodal (single session) | `--vision-device` | first request of either kind | the text value | as above |

- The idle check runs every 5 seconds. An encoder unloads when its last request finished more
  than its idle time ago and no request uses it.
- A request for an unloaded encoder loads it first. Concurrent requests wait for the one load.
- `GetModelInfo` answers from the model manifest. It does not load an encoder.
- `GetRuntimeStatus` reports, for each encoder: loaded or not, device, and the time since the
  last use.
- An encoder load failure returns the ORT or Core ML error to the request. The encoder stays
  unloaded. The next request tries to load it again.

### 2.4 Setting

- Location: the "Local Content Recognition" settings page (`SemanticGenerationSettingsPanel.qml`),
  in a new row "Keep AI model loaded".
- Control: a combo box with the five values of Section 1.2, item 4.
- `QSettings` key: `ai/sidecarIdleStopSeconds`. Values: `0`, `60`, `300`, `900`, `-1`. Default
  `300`. An unknown value reads as `300`.
- A change applies at once to the process idle time of the running service. The text encoder
  idle time applies at the next process start. The row shows the help text "Applies to the
  sidecar the next time it starts." only for that part. The help text is a plain label.

### 2.5 Prewarm

- `SearchController::PrewarmNaturalLanguageSearch()` runs when the search palette opens and when
  the natural-language switch turns on.
- It does nothing when the switch is off, when no semantic provider exists, or when the neutral
  text embedding is already in the cache.
- Otherwise it runs one background job: acquire a lease, start the runtime, embed the neutral
  text `a photo` into the query embedding cache, release the lease. The release starts the idle
  stop timer.
- A prewarm failure writes a warning to the log. It shows no message. The user sees the real
  error when they press Enter.

## 3. Scope

### 3.1 Included work by module

| Module | Responsibility |
| --- | --- |
| `AiSidecarRuntimeService` | Idle stop timer; asynchronous idle stop; restart rules; new process arguments |
| `ProjectService` | Reads the setting; applies it to the service; prewarm entry of the provider |
| `ProjectSemanticSearchProvider` | Uses the interactive start from the worker thread; prewarm |
| `ImageAnalysisEnvironment` | Passes the provider config change to the service |
| `SearchController`, palette QML | Prewarm call |
| `SemanticGenerationSettingsPanel.qml` | The setting row |
| Rust engine and server | Lazy encoder load; idle unload; text device; status fields |

### 3.2 Exclusions

- Model download and model activation flows. They keep their current lease use.
- macOS Core ML device choice for the image encoder. It keeps the current `--device` rule.
- A memory pressure signal from the operating system.

## 4. Current source audit

All C++ paths are under `alcedo_studio/src`. All Rust paths are under `rust/puerh_mind/src`.

| Area | Current owner and path | Current behavior | Required change |
| --- | --- | --- | --- |
| Lease release | `AiSidecarRuntimeService::ReleaseLease`, `app/ai_sidecar_runtime_service.cpp:468-480` | At 0 leases, `Stop()` runs at once. | Idle stop timer. |
| Stop | `AiSidecarRuntimeService::Stop`, same file :424-456 | Terminate, wait up to 1 s, kill, wait up to 1 s, on the owner (GUI) thread. | Asynchronous variant for the idle stop. |
| Start with a running process | `StartAndWaitBody`, same file :319-404 | Reuses the process for any options. Test `RunningSidecarReusesExistingProcessForDifferentOptions` asserts this. | Restart rules of Section 2.2. |
| Search start | `ProjectSemanticSearchProvider::Search`, `app/project_service.cpp:196-246` | Local lease; `StartAndWait` from the query worker. The call runs the boot on the GUI thread through `BlockingQueuedConnection` and blocks the GUI thread for the whole boot (up to 60 s). | Use `StartAndWaitInteractive`, which keeps the GUI event loop running during the boot. |
| Provider config change | `image_analysis_environment.cpp:215-222` | `provider_configs_dirty` is ignored. | Restart rule. |
| Device | `AiSidecarRuntimeOptions::device`, `include/app/ai_sidecar_runtime_service.hpp:57-77` | No production code sets it; the sidecar gets `--device auto` (DirectML preferred on Windows, Core ML on macOS). | Add `text_device` (`cpu`) and `vision_device` (`device`). |
| Rust start | `main.rs:21-90`, `service/inference.rs:14-33` | `build_semantic_engine` loads the model before the gRPC server starts. Ping is not served until the load is done. | No encoder load at start. |
| Encoder sessions | `OrtClipEngine::new`, `service/ort_clip.rs:817-919`; `CoreMlClipEngine::new`, `service/coreml_clip.rs:62-139` | Both encoders load at construction and stay for the process lifetime. | Lazy load and idle unload. |
| Engine interface | `EmbeddingEngine`, `service/embedding.rs:19-37` | No load or unload operation. | Residency status. |
| Status | `server/semantic.rs:580-625`; `proto/semantic.proto:150-173` | `GetModelInfo` fails when the engine is not ready. `GetRuntimeStatus` has state, provider, batch values, uptime. | Model info from the manifest; encoder residency fields. |
| Settings | none | No key for idle time or device. | `ai/sidecarIdleStopSeconds`. |

Current tests:

- `AiSidecarRuntimeServiceTest` (`alcedo_studio/tests/app/ai_sidecar_runtime_service_test.cpp`)
  with the fake runtime `AiSidecarFakeRuntime`: 19 tests, including
  `LeasesStopRuntimeOnlyAfterLastRelease`, `RunningSidecarReusesExistingProcessForDifferentOptions`,
  `StopKillsHungRuntime`, `RuntimeArgumentsCarryModelAndDeviceConfiguration`.
- Rust: `ort_clip.rs` model tests (skip without `ALCEDO_MIND_TEST_MODEL_ROOT`), device parse
  tests, `server/semantic.rs` `reports_model_info_and_runtime_status`, `config.rs` tests.

## 5. Target architecture

### 5.1 Owners

| Owner | Input | Output | Changes | Lifetime | Errors |
| --- | --- | --- | --- | --- | --- |
| `AiSidecarRuntimeService` | leases, start options, idle delay, config change mark | process state | process, timer, lease count | per `ProjectService` | status issue and message |
| `EncoderResidency<T>` (proposed, Rust) | loader closure, idle time | a loaded session for one request | its session slot and last use time | per engine | loader error |
| `OrtClipEngine`, `CoreMlClipEngine` | requests | embeddings | residency slots | per process | load and run errors |

`EncoderResidency<T>` holds `Mutex<Option<T>>` and the last use time. `with_session(f)` locks the
mutex, loads the session when the slot is empty, runs `f`, and records the time. `unload_if_idle`
locks the mutex and drops the session when it is idle. The mutex already serializes a request and
an unload, so no other mechanism is needed.

### 5.2 Required interleavings

The plan adds no version or token. It handles these real interleavings with the current owner
thread and the process state:

1. **Idle stop and a new start.** The idle timer expires. The service sends terminate and sets
   `kStopping`. Before the process exits, a search on the query worker calls
   `StartAndWaitInteractive`. Today `IsRunning()` is true for `kStopping`, so the start waits for
   readiness of a dying process and fails. Rule: a start in `kStopping` waits for the process to
   exit (up to the kill time), then starts a new process. Test
   `StartDuringIdleStopWaitsAndStartsNewProcess`.
2. **Model change within the idle time.** The user activates another model. A search starts
   within the idle time. The running process has the old model, and the provider reports the
   model mismatch. Rule: Section 2.2. Test `ModelIdentityChangeRestartsProcessWithoutOtherLease`.

### 5.3 Primary success call chain (search after prewarm)

```text
palette opens -> SearchController::PrewarmNaturalLanguageSearch -> background job
  -> provider: lease -> StartAndWaitInteractive (owner thread pumps events)
  -> sidecar starts (no encoder load) -> Ping + GetModelInfo ready in < 1 s
  -> EmbedText("a photo") -> text encoder loads on CPU -> cache entry -> lease released
  -> idle stop timer starts (300 s)
user Enter -> provider: cache miss for the query -> lease (timer stops) -> runtime ready
  -> EmbedText(query) on the loaded text encoder -> lease released -> timer starts
```

### 5.4 Primary failure chain

```text
EmbedText -> text encoder load fails (bad file) -> gRPC error with the ORT message
  -> provider throws -> Library shows "Library query failed: <message>" with Retry
  -> the process stays; the encoder slot stays empty; Retry loads again
```

## 6. File and API map

### 6.1 Current files that change

| File | Phase | Change |
| --- | --- | --- |
| `include/app/ai_sidecar_runtime_service.hpp`, `app/ai_sidecar_runtime_service.cpp` | SR1, SR2 | Idle stop; async idle stop; restart rules; new options and arguments |
| `include/app/project_service.hpp`, `app/project_service.cpp` | SR1, SR3 | Setting read; interactive start in search; prewarm |
| `ui/alcedo_main/album_backend/image_analysis_environment.cpp` | SR1 | Pass the config change mark |
| `alcedo_studio/tests/app/ai_sidecar_runtime_service_test.cpp`, `ai_sidecar_fake_runtime_main.cpp` | SR1, SR2 | New tests; fake runtime accepts the new arguments |
| `rust/puerh_mind/src/service/ort_clip.rs`, `coreml_clip.rs`, `inference.rs`, `embedding.rs`, `config.rs`, `main.rs` | SR2 | Lazy load, idle unload, text device |
| `rust/puerh_mind/src/server/semantic.rs`, `rust/puerh_mind/proto/semantic.proto` | SR2 | Model info from manifest; residency status fields (new field numbers only) |
| `include/ui/alcedo_main/album_backend/search_controller.hpp`, `.../search_controller.cpp` | SR3 | `PrewarmNaturalLanguageSearch` |
| `ui/alcedo_main/qml/SearchPalette.qml` | SR3 | Prewarm call on open and on switch on |
| `ui/alcedo_main/qml/SemanticGenerationSettingsPanel.qml` | SR3 | Setting row |

### 6.2 Proposed files

| File | Phase | Responsibility |
| --- | --- | --- |
| `rust/puerh_mind/src/service/encoder_residency.rs` | SR2 | `EncoderResidency<T>` and its unit tests |

### 6.3 Proposed APIs

| API | Input | Output | Validation | Error | Changes state |
| --- | --- | --- | --- | --- | --- |
| `AiSidecarRuntimeService::SetIdleStopDelay(std::chrono::seconds)` | delay; negative means no timer | none | none | none | Yes (applies to the next release; a running timer restarts with the new value) |
| `AiSidecarRuntimeService::MarkProviderConfigsChanged()` | none | none | none | none | Yes (restart flag) |
| `AiSidecarRuntimeOptions::text_device`, `vision_device`, `text_idle_unload`, `vision_idle_unload` | values | arguments | parsed by the sidecar | sidecar start failure with the parse error | No |
| `ProjectService::SetAiSidecarIdleStopSeconds(int)` | setting value | none | maps unknown values to 300 | none | Yes |
| `SemanticSearchProvider::Prewarm()` | none | none | none | exception, logged by the caller | Cache only |
| `SearchController::PrewarmNaturalLanguageSearch()` (`Q_INVOKABLE`) | none | none | switch and provider checks | logged | No |
| Rust `EncoderResidency::with_session`, `unload_if_idle`, `status` | closure, now | result, bool, status | none | loader error | Yes |

## 7. Repository rules that affect this work

Read `AGENTS.md` and the applicable skills again before each phase.

- No fallback without authorization. An encoder load failure on the configured device returns
  that error. The sidecar does not switch devices. The text encoder device is a fixed policy
  (Section 1.2, item 6), not a fallback.
- Require an executable interleaving before a consistency mechanism. Section 5.2 lists the two
  interleavings. Add nothing else.
- Rust tests: run `cargo test -- --skip live_` for offline tests. Rebuild the release
  `alcedo_mind.exe` before a live test from C++.
- MSVC builds through the PowerShell tool. Tests use the debug build. Do not run the full suite.
- Terminology: do not use the prohibited terms of `AGENTS.md`.
- LF line endings. Some Rust files had CRLF problems after Python writes on Windows; check
  `git diff --stat` after a scripted edit.

## 8. Phase summary

| Phase | Result | Main modules | Dependency | Expected diff | Status |
| --- | --- | --- | --- | ---: | --- |
| SR1 | Process idle stop; no GUI block on search start or idle stop; restart rules | `AiSidecarRuntimeService`, `ProjectService`, image analysis environment | none | 700–1100 | planned |
| SR2 | Lazy encoder load, idle unload, text encoder on CPU | Rust engine and server; runtime options | SR1 | 900–1500 | planned |
| SR3 | Setting row, prewarm, and resource measurements | settings QML, `SearchController`, provider | SR1, SR2 | 400–700 | planned |

## 9. Detailed phases

### Phase SR1 — Process idle stop and restart rules

**Objective and deliverables**

- The process stays alive for the idle delay after the last lease.
- The idle stop does not block the GUI thread.
- A search start from the query worker does not block the GUI thread.
- The restart rules of Section 2.2 apply.
- The baseline measurement of Step 1 is recorded.

**Inputs and prerequisites**

- Source at or after `ada6f002c`.

**Modules, files, and APIs**

- `AiSidecarRuntimeService`: `SetIdleStopDelay`, `MarkProviderConfigsChanged`, private
  `idle_stop_timer_` (`QTimer`, single shot, owner thread), private `BeginIdleStop()`.
- `ProjectService`: read `ai/sidecarIdleStopSeconds` when it creates the service and call
  `SetIdleStopDelay`. Add `SetAiSidecarIdleStopSeconds`.
- `ProjectSemanticSearchProvider::Search`: `StartAndWaitInteractive` in place of
  `StartAndWait`.
- `image_analysis_environment.cpp`: call `MarkProviderConfigsChanged` when
  `provider_configs_dirty` is true.

**Data rules and invariants**

- The timer runs only when the lease count is 0 and the process runs.
- `BeginIdleStop` runs only on the owner thread. It sets `kStopping`, calls `terminate()`, and
  starts a kill timer of `graceful_stop_timeout`. The `QProcess::finished` handler completes the
  stop with the same state changes as `Stop()`.
- `Stop()` (blocking) stays for project close and the destructor.
- A start in `kStopping` waits for the exit, then starts (Section 5.2, item 1).
- Restart for model identity compares `model_root`, `model_id`, and `revision` of the running
  options with the request.

**Implementation steps**

1. Baseline measurement. With the current `main` release build and a project with embeddings,
   measure: (a) time from Enter to the first result for a natural-language query, cold and
   repeated; (b) working set of `alcedo_mind.exe` after the start; (c) dedicated GPU memory of
   `alcedo_mind.exe` in Task Manager; (d) time to `kReady`. Record the values in the completion
   record.
2. Add the timer and `SetIdleStopDelay`. Change `ReleaseLease` and `AcquireLease`.
3. Add `BeginIdleStop` and the `finished` handler. Keep `Stop()` for project close.
4. Handle `kStopping` in `StartAndWaitBody`.
5. Add the restart rules in `StartAndWaitBody`.
6. Change `ProjectSemanticSearchProvider::Search` to `StartAndWaitInteractive`.
7. Wire the setting read in `ProjectService` and the config change mark in the image analysis
   environment.

**Primary success call chain**

```text
search lease released -> ReleaseLease (count 0) -> idle_stop_timer_ start (300 s)
next search within 300 s -> AcquireLease -> timer stop -> StartAndWaitInteractive -> running, ready
timer expires -> BeginIdleStop -> terminate -> finished -> kStopped
```

**Primary failure and restore call chain**

```text
BeginIdleStop -> process ignores terminate -> kill timer -> kill -> finished -> kStopped
kill does not end the process -> kFailed / kStopTimedOut (same as Stop())
```

**Tests and evidence** (target `AiSidecarRuntimeServiceTest`, fake runtime)

| Test | Assertion |
| --- | --- |
| `LeasesStopRuntimeOnlyAfterLastRelease` (kept, delay 0) | State is `kStopped` right after the last release. |
| `IdleDelayKeepsRuntimeRunningAfterLastRelease` | With delay 2 s, state is `kReady` 200 ms after the release, and the process id is unchanged. |
| `IdleDelayStopsRuntimeWhenNoLeaseReturns` | With delay 300 ms, state becomes `kStopped` within 3 s. |
| `NewLeaseCancelsIdleStop` | Release, then a lease after 100 ms with delay 300 ms: state stays `kReady` after 1 s. |
| `NegativeDelayKeepsRuntimeUntilProjectClose` | No stop after 1 s; `StopForProjectClose` stops it. |
| `IdleStopDoesNotBlockOwnerThread` | With a runtime that ignores terminate, a queued call on the owner thread runs within 50 ms after the timer expires. |
| `StartDuringIdleStopWaitsAndStartsNewProcess` | A start during `kStopping` returns true with a new process id. |
| `ModelIdentityChangeRestartsProcessWithoutOtherLease` | A start with another `model_id` and only the caller's lease gives a new process id. |
| `RunningSidecarReusesExistingProcessForDifferentOptions` (changed) | With a second lease held, a start with another `model_id` keeps the process id. |
| `ProviderConfigChangeRestartsProcessAtNextStart` | After `MarkProviderConfigsChanged`, the next start without other leases gives a new process id. |
| `WorkerStartKeepsOwnerThreadResponsive` | During a start from a worker thread with a slow fake boot, a queued call on the owner thread runs within 50 ms. |

**Build and run commands**

```powershell
Set-Location D:\Projects\pu-erh_lab
cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 4 --target AiSidecarRuntimeServiceTest AiSidecarFakeRuntime ImageAnalysisControllerTest
$env:PATH = "$PWD\build\debug\vcpkg_installed\x64-windows\debug\bin;$env:PATH"
ctest --test-dir build/debug -R "AiSidecarRuntimeServiceTest|ImageAnalysisControllerTest|ImageAnalysisServiceTest" --output-on-failure
```

Put logs under `build/tmp/sidecar_residency_sr1/`.

**Exit criteria**

- [ ] All named tests pass.
- [ ] Baseline measurements are recorded.
- [ ] Manual check: two natural-language searches within 5 minutes start the sidecar once (log
      `semantic.runtime` start lines).

**Expected diff**

700–1100 lines (service 250–400, project service and environment 50–100, tests 350–550, fake
runtime 30–60).

**Completion record**

```text
(empty — fill when the phase is complete; use Section 13)
```

### Phase SR2 — Lazy encoder load and idle unload

**Objective and deliverables**

- The sidecar starts without an encoder load. Ping and `GetModelInfo` answer at once.
- Each encoder loads on its first request and unloads after its idle time.
- The text encoder runs on the CPU.
- `GetRuntimeStatus` reports encoder residency.
- The C++ runtime options pass the new arguments.

**Inputs and prerequisites**

- Phase SR1 complete.

**Modules, files, and APIs**

- Proposed `service/encoder_residency.rs`: `EncoderResidency<T>`.
- `OrtClipEngine`: store the model paths and the session builders, not the sessions. Use one
  `EncoderResidency` for text, one for image, or one for a multimodal session.
- `CoreMlClipEngine`: the same rule for its two models.
- `inference.rs`: `build_semantic_engine` validates the manifest and the files only.
- `config.rs`: `--text-device` (default `cpu`), `--vision-device` (default the `--device` value),
  `--text-idle-unload-seconds` (default 300; `-1` never; `0` after each request),
  `--vision-idle-unload-seconds` (default 60).
- `main.rs`: a Tokio interval task (5 s) calls `unload_if_idle` for each slot.
- `semantic.proto`: new fields in `GetRuntimeStatusResponse` for each encoder: `loaded`,
  `device`, `idle_ms`. Use new field numbers only.
- `server/semantic.rs`: `GetModelInfo` from the manifest.
- C++ `AiSidecarRuntimeOptions` and `BuildArguments`: the four new values. `ProjectService` sets
  `text_idle_unload` from the setting (`-1` for "while the project is open").

**Data rules and invariants**

- A request holds the slot mutex while the session runs. An unload takes the same mutex.
- An encoder load failure leaves the slot empty and returns the error.
- A batch of image requests counts as use at the end of each batch.
- The model info fields do not change (same values as today for each profile).

**Implementation steps**

1. Write `EncoderResidency<T>` with an injectable clock for tests.
2. Move the session creation of `OrtClipEngine` into loader closures. Use the text device for the
   text loader and the vision device for the image loader.
3. Do the same for `CoreMlClipEngine`.
4. Change `build_semantic_engine` and `GetModelInfo`.
5. Add the configuration flags and the interval task.
6. Add the status fields to the proto and the server.
7. Add the C++ options, the arguments, and the fake runtime argument parse.
8. Rebuild the release `alcedo_mind.exe` for the live checks.

**Primary success call chain**

```text
process start -> manifest check -> gRPC serve -> Ping ok -> GetModelInfo ok (no load)
EmbedText -> text slot empty -> load on cpu -> run -> last use = now
interval task (5 s) -> text idle > 300 s -> drop session -> status loaded = false
```

**Primary failure and restore call chain**

```text
EmbedImageBatch -> image slot load on directml fails -> gRPC INTERNAL with the ORT message
  -> slot stays empty -> the generation task reports the error -> next request loads again
```

**Tests and evidence**

| Test | Assertion |
| --- | --- |
| `residency_loads_once_for_concurrent_requests` (Rust unit) | Two threads call `with_session`; the loader runs once. |
| `residency_unloads_after_idle_time` (Rust unit) | With a fake clock past the idle time, `unload_if_idle` drops the session and status reports not loaded. |
| `residency_keeps_session_while_a_request_runs` (Rust unit) | `unload_if_idle` during a running `with_session` does not drop the session. |
| `residency_failed_load_leaves_slot_empty` (Rust unit) | A failing loader returns its error; the next call runs the loader again. |
| `negative_idle_time_never_unloads` (Rust unit) | Idle time `-1` keeps the session after any clock advance. |
| `model_info_is_served_before_any_encoder_loads` (Rust, test model) | `GetModelInfo` returns the expected dimension; status reports both encoders not loaded. |
| `text_request_loads_only_the_text_encoder` (Rust, test model) | After `EmbedText`, status reports text loaded and image not loaded. |
| `parses_text_and_vision_device_flags` (Rust `config.rs`) | The flags parse; `cuda` and `metal` are still rejected. |
| `RuntimeArgumentsCarryModelAndDeviceConfiguration` (C++, extended) | The arguments contain the four new values. |
| `embeds_text_with_ort_model` and the other model tests (existing) | Still pass with lazy load. |

**Build and run commands**

```powershell
Set-Location D:\Projects\pu-erh_lab\rust\puerh_mind
cargo test -- --skip live_
Set-Location D:\Projects\pu-erh_lab
cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 4 --target AiSidecarRuntimeServiceTest AiSidecarFakeRuntime
$env:PATH = "$PWD\build\debug\vcpkg_installed\x64-windows\debug\bin;$env:PATH"
ctest --test-dir build/debug -R "AiSidecarRuntimeServiceTest" --output-on-failure
```

Run the model tests with `ALCEDO_MIND_TEST_MODEL_ROOT` set. Record them as skipped when the
variable is absent.

**Exit criteria**

- [ ] All offline Rust tests and the named C++ tests pass.
- [ ] The model tests pass with a local model, or are recorded as skipped with the reason.
- [ ] Manual check with the release sidecar: an image analysis run with a remote provider shows
      both encoders not loaded in `GetRuntimeStatus` (log line).

**Expected diff**

900–1500 lines (residency 200–300 with tests, ORT engine 250–400, Core ML engine 100–200, server
and proto 80–150, config 60–100, C++ 60–120, tests 150–300).

**Completion record**

```text
(empty — fill when the phase is complete; use Section 13)
```

### Phase SR3 — Setting, prewarm, and measurements

**Objective and deliverables**

- The setting row of Section 2.4.
- The prewarm of Section 2.5.
- The final measurements, compared with the SR1 baseline.

**Inputs and prerequisites**

- Phases SR1 and SR2 complete.
- Phase GS1 of the [Global Search Redesign Plan](../ui/global_search_redesign_plan.md)
  complete, because the prewarm fills the query embedding cache of GS1. If GS1 is not complete,
  stop and wait.

**Modules, files, and APIs**

- `SemanticGenerationSettingsPanel.qml`: the row. Use the current settings controls and
  `appTheme` tokens. Do not use Material controls.
- `ProjectService::SetAiSidecarIdleStopSeconds`; `SemanticSearchProvider::Prewarm`.
- `SearchController::PrewarmNaturalLanguageSearch`: runs the job on `SearchRequestWorker` with a
  new kind `kPrewarm`, so it does not replace a preview or apply request.
- The search palette QML in use: call the prewarm in `onOpened` and when the switch turns on.

**Data rules and invariants**

- The prewarm holds a lease only during its job.
- The prewarm does nothing when the neutral embedding is in the cache.
- A setting change calls `SetIdleStopDelay` at once.

**Implementation steps**

1. Add the setting row and the `QSettings` read and write. Call
   `ProjectService::SetAiSidecarIdleStopSeconds` on change.
2. Add `Prewarm` to the provider and `PrewarmNaturalLanguageSearch` to the controller.
3. Add the QML calls.
4. Final measurement: repeat SR1 Step 1. Also measure (e) working set with only the text encoder
   loaded, (f) working set with both encoders loaded, (g) time from palette open to a ready text
   encoder.

**Primary success call chain**

Section 5.3.

**Primary failure and restore call chain**

```text
Prewarm -> runtime start fails -> warning in the log -> no cache entry, no UI change
user Enter -> the real start error appears in the Library error row
```

**Tests and evidence**

| Test | Assertion |
| --- | --- |
| `PrewarmEmbedsNeutralTextOnce` (`AlbumBackendSearchWorkerTest`, fake provider) | Two prewarm calls call `Prewarm` once. |
| `PrewarmDoesNothingWhenSwitchIsOff` (`AlbumBackendSearchWorkerTest`) | No provider call. |
| `PrewarmDoesNotReplacePreviewRequest` (`SearchRequestWorkerTest`) | A pending preview and a pending prewarm both run. |
| `IdleStopSettingAppliesToRunningService` (`AiSidecarRuntimeServiceTest` or project service test) | A setting change from 300 to 0 while no lease is held stops the process. |
| `UnknownIdleStopSettingReadsAsDefault` | The value 42 reads as 300. |

**Build and run commands**

```powershell
Set-Location D:\Projects\pu-erh_lab
cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 4 --target AlbumBackendSearchWorkerTest SearchRequestWorkerTest AiSidecarRuntimeServiceTest alcedo_main
$env:PATH = "$PWD\build\debug\vcpkg_installed\x64-windows\debug\bin;$env:PATH"
ctest --test-dir build/debug -R "AlbumBackendSearchWorkerTest|SearchRequestWorkerTest|AiSidecarRuntimeServiceTest" --output-on-failure
```

**Exit criteria**

- [ ] All named tests pass.
- [ ] The measurement table has the SR1 baseline and the SR3 values for items (a) to (g).
- [ ] Manual check: open the palette with natural-language search on, wait 3 s, press Enter;
      the log shows no sidecar start after Enter.

**Expected diff**

400–700 lines.

**Completion record**

```text
(empty — fill when the phase is complete; use Section 13)
```

## 10. Cross-phase acceptance matrix

| Behavior | Expected result | Phase |
| --- | --- | --- |
| Two searches within the idle time | One sidecar start; the second search has no load time. | SR1 |
| No use for longer than the idle time | The process stops; the GUI does not pause. | SR1 |
| Setting "unload after use" | Current behavior: stop at the last release. | SR1, SR3 |
| Setting "while the project is open" | The process stops only at project close. | SR1, SR3 |
| Model change within the idle time | The next operation runs on the new model. | SR1 |
| Provider config edit within the idle time | The next image analysis uses the new config. | SR1 |
| Image analysis with a remote provider | No CLIP encoder loads. | SR2 |
| Embedding generation, then 60 s idle | The image encoder unloads; the text encoder stays. | SR2 |
| Cold search start | The GUI stays responsive during the boot. | SR1 |
| Palette open with natural-language search on | The text encoder is ready before Enter. | SR3 |
| Encoder load failure | The request returns the error; the next request loads again. | SR2 |

## 11. Build and evidence rules

- Presets: `win_debug` for C++ tests; `cargo test` in `rust/puerh_mind` for Rust; release
  `alcedo_mind.exe` and `alcedo_main` for measurements.
- macOS: Core ML engine changes need a `macos_debug` build with tests on, or the item is recorded
  as not run on macOS.
- Evidence: `build/tmp/sidecar_residency_<phase>/`. Remove it when the phase is complete.
- Record each test as passed, failed, skipped, or not run. A skipped test is not a pass.

## 12. Risks and stop conditions

| Risk | Signal | Response |
| --- | --- | --- |
| CPU text encoding is slow on a large text model | SR3 measurement (a) warm above 150 ms | Stop. Report the value. The device rule changes only with a user decision. |
| Memory of an idle process is high even with no encoder | SR3 measurement (b) | Report. The idle delay setting lets the user choose. |
| `QProcess::finished` is not delivered when the owner thread is busy | `IdleStopDoesNotBlockOwnerThread` fails | Fix the handler. Do not return to the blocking stop for the idle path. |
| Lazy load hides a bad model file until first use | Model test with a broken file | Expected behavior: the first request reports the error. Record it in the completion record. |
| Core ML engine cannot unload and load again on one process | macOS test | Stop and update the plan for Core ML. |

Stop implementation when a required owner API does not exist, when the source audit no longer
matches the branch, when a phase can pass 2000 changed lines, or when an operation needs a
fallback that the user did not authorize. Update this plan before work continues.

## 13. Completion record template

```text
Phase / date / status:
Source revision and branch:
Actual changed modules:
Implemented behavior:
Explicitly unimplemented items:
Primary success call chain:
Primary failure and restore call chain:
Build and test commands with exit codes:
Discovered / passed / failed / skipped counts:
Manual verification:
Measurements (SR1 baseline and SR3 final):
Evidence path:
Remaining defects or unavailable platforms:
```
