---
name: raw-processor-module
description: Use when modifying the RAW Processor module in pu-erh_lab, especially RawProcessor, RAW decode GPU operators, or Metal RAW shaders and their CMake wiring.
---

# Raw Processor Module

Use this skill when working under `pu-erh_lab/src/decoders/processor/` for RAW decode and demosaic work.

## Workflow

- Keep RAW pipeline entrypoint changes in `pu-erh_lab/src/decoders/processor/raw_processor.cpp`.
- Keep RAW GPU operator code under `pu-erh_lab/src/decoders/processor/operators/gpu/`.
- For Metal implementations in the RAW Processor module, place shader sources in `pu-erh_lab/src/decoders/processor/operators/gpu/metal_shader/`.
- When adding or renaming a RAW Processor Metal shader, update `pu-erh_lab/src/CMakeLists.txt` so the `.metal` file is compiled to `.air`, linked to `.metallib`, added to `RawProcessorOpMetalShaders`, and exposed to the matching C++ source via `target_compile_definitions(...)`.

## Rules

- Match RAW Metal operator behavior to the corresponding CPU or CUDA implementation before changing pipeline flow.
- Prefer adding dedicated RAW operator entrypoints instead of putting Metal shader dispatch directly into `raw_processor.cpp`.
- If a RAW Metal operator changes output format or dimensions, update the RAW Processor integration and any RAW-stage tests in the same change.
