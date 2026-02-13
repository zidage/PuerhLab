# Pu-erh Lab

<p align="right"><a href="./README.md"><strong>English</strong></a> | <a href="./README.zh-CN.md">简体中文</a></p>

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/License-Apache_2.0-blue)
![Stage](https://img.shields.io/badge/stage-pre--alpha-orange)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue)

Pu-erh Lab is an open-source RAW photo processor and digital asset management (DAM) project. The current codebase focuses on a high-performance, non-destructive image pipeline and a service-oriented architecture for album/project workflows.

> Project stage: **Pre-Alpha**. APIs and behavior may change as modules are still being actively refactored.

## Early Demo

Video: [BiliBili](https://www.bilibili.com/video/BV1bPcxzzEeM)

<table>
  <colgroup>
    <col style="width: 80%" />
    <col style="width: 20%" />
  </colgroup>
  <tbody>
    <tr>
      <td><img src="docs/demo/1.png" alt="Modern UI interface" width="100%" /></td>
      <td>Modern UI Interface</td>
    </tr>
    <tr>
      <td><img src="docs/demo/2.png" alt="Highlight transition" width="100%" /></td>
      <td>Film-like Highlight Transition</td>
    </tr>
    <tr>
      <td><img src="docs/demo/3.png" alt="LUT and history stack" width="100%" /></td>
      <td>LUT Support / Unlimited History Stack / Git-like Version Control</td>
    </tr>
    <tr>
      <td><img src="docs/demo/4.png" alt="Export workflow" width="100%" /></td>
      <td>Advanced Exporting Feature</td>
    </tr>
  </tbody>
</table>

## Vision

Pu-erh Lab aims to provide a professional workflow for photographers by combining:

- robust digital asset management for large libraries,
- a non-destructive, stage-based editing pipeline,
- and a responsive UI experience designed for iterative image development.

## Key Technical Features

### High-Performance Core

- Concurrency-first architecture with tile-based rendering and task scheduling.
- Modern C++20 codebase with modular libraries.
- Optional CUDA acceleration path for RAW processing and pipeline stages.

### Professional Imaging Pipeline

- 32-bit float-oriented editing workflow for preview and export stages.
- LibRaw-based RAW decoding with CPU/GPU operator split (WIP).
- Color management stack using OpenColorIO + lcms2, with LUT support (`.cube`) (WIP).
- Non-destructive operation graph with history/version capabilities.

### Asset Management ("Sleeve" System)

- Custom Sleeve abstraction for folder/file organization and large-library navigation.
- DuckDB-backed metadata/index storage through mapper/service layers.
- Application services for import, filtering, thumbnailing, pipeline management, and export.

## Recent Progress

Recent updates across the codebase include:

- Geometry workflow: interactive crop/rotate editing and integration of `CROP_ROTATE` in the pipeline.
- Color/tone workflow: expanded HLS adjustment model and curve editing improvements.
- Color output path: ODT/LUT resource caching and lifecycle improvements for better runtime behavior.
- Preview and memory: preview quality/performance tuning and VRAM usage optimization.
- DAM/app layer: stronger project/folder management and ongoing migration to service-level workflows.
- History/version workflow: history stack improvements, sync fixes, and album reload flow updates.
- Export workflow: export progress reporting and stability fixes.

## Build from Source (Windows Only)

This section is intentionally Windows-focused and mirrors the current build setup in `CMakeLists.txt` and `CMakePresets.json`.

### 1) Prerequisites

- Windows 10/11 x64
- Visual Studio 2022 (MSVC toolchain, x64)
- CMake 3.21+
- Ninja
- Qt 6 (MSVC 2022 x64), including modules used by the project
- Git
- NVIDIA CUDA Toolkit (enables CUDA targets automatically when found)

### 2) Dependency Resolution in CMake

The top-level CMake currently requires/resolves:

- Qt6: `Widgets`, `Quick`, `OpenGL`, `OpenGLWidgets` (plus QML demo modules in `tests/gui_pocs/album_editor_qml`)
- Core libs: `OpenCV`, `Eigen3`, `OpenGL`, `glad`, `hwy`, `lcms2`, `OpenColorIO`, `OpenImageIO`, `libraw`, `xxHash`
- Parallelism/testing: `OpenMP`, `googletest` (via `FetchContent`)
- Windows-specific profiling: `easy_profiler`
- Local third-party imported binaries: DuckDB from `pu-erh_lab/third_party/libduckdb-windows` (fallback Exiv2 path is also defined)

### 3) Configure and Build

Clone and initialize submodules:

```powershell
git clone --recursive https://github.com/zidage/PuerhLab.git
cd PuerhLab
```

Bootstrap local vcpkg if needed:

```powershell
.\vcpkg\bootstrap-vcpkg.bat
```

Recommended: use `scripts/msvc_env.cmd` so MSVC environment variables are set automatically.
Adjust the Qt/easy_profiler paths below to your local environment.

```powershell
# Debug configure
.\scripts\msvc_env.cmd --preset win_debug `
  -DCMAKE_PREFIX_PATH="D:/Qt/6.9.3/msvc2022_64/lib/cmake" `
  -Deasy_profiler_DIR="$PWD/pu-erh_lab/third_party/easy_profiler-v2.1.0-msvc15-win64/lib/cmake/easy_profiler"

# Debug build
.\scripts\msvc_env.cmd --build build/debug --parallel
```

```powershell
# Release configure
.\scripts\msvc_env.cmd --preset win_release `
  -DCMAKE_PREFIX_PATH="D:/Qt/6.9.3/msvc2022_64/lib/cmake" `
  -Deasy_profiler_DIR="$PWD/pu-erh_lab/third_party/easy_profiler-v2.1.0-msvc15-win64/lib/cmake/easy_profiler" `
  -DPUERHLAB_WINDEPLOYQT="D:/Qt/6.9.3/msvc2022_64/bin/windeployqt.exe"

# Release build + install
.\scripts\msvc_env.cmd --build build/release --parallel
.\scripts\msvc_env.cmd --install build/release --prefix build/install
```

Create a ZIP package (CPack):

```powershell
cpack --config build/release/CPackConfig.cmake
```

### 4) Run Demo Binaries

Common demo executables after a Debug build:

```powershell
.\build\debug\pu-erh_lab\tests\CompleteUIDemo.exe
.\build\debug\pu-erh_lab\tests\ThumbnailAlbumQtDemo.exe
.\build\debug\pu-erh_lab\tests\ImagePreview.exe
.\build\debug\pu-erh_lab\tests\gui_pocs\album_editor_qml\AlbumEditorQmlDemo.exe
```

### 5) Tests and Dev Utilities

- Several tests are currently runnable as standalone executables under `build\debug\pu-erh_lab\tests\`.
- During app-layer refactoring, some historical unit tests are intentionally disabled in `pu-erh_lab/tests/CMakeLists.txt`.
- Formatting/lint targets are available (clang-tidy is not fully set up yet):

```powershell
.\scripts\msvc_env.cmd --build build/debug --target format
.\scripts\msvc_env.cmd --build build/debug --target tidy
```

## Roadmap

Roadmap and ongoing milestones:

- [docs/roadmap/roadmap.md](docs/roadmap/roadmap.md)

## License

Apache-2.0. See [LICENSE](LICENSE).



