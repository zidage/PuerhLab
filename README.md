# Pu-erh Lab

<p align="right"><a href="./README.md"><strong>English</strong></a> | <a href="./README.zh-CN.md">简体中文</a></p>

![License](https://img.shields.io/badge/License-GPLv3-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900)
![C++](https://img.shields.io/badge/C++-20-blue)

**Pu-erh Lab** is an open-source RAW photo editor and digital asset management (DAM) project. It is designed to provide a new choice to photographers who seek a lightweight, high-performance, and largely industry-compatible workflow for their photo editing and library management needs. 

>Pu-erh Lab is _**NOT an alternative**_ to the existing commercial software nor other open-source projects.


## Early Demo

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
      <td>Advanced Adjustment Controls</td>
    </tr>
    <tr>
      <td><img src="docs/demo/3.png" alt="LUT and history stack" width="100%" /></td>
      <td>LUT Support / Unlimited History Stack / Git-like Version Control</td>
    </tr>
    <tr>
      <td><img src="docs/demo/5.png" alt="LUT and history stack" width="100%" /></td>
      <td>Lens Correction Support</td>
    </tr>
    <tr>
      <td><img src="docs/demo/4.png" alt="Export workflow" width="100%" /></td>
      <td>Advanced Exporting Feature</td>
    </tr>
  </tbody>
</table>

## Key Technical Features

### High-Performance Core

- CUDA-accelerated image processing pipeline with the highest real-time preview resolution running at **60 FPS** on modern GPUs with large RAW files (e.g., 45MP).
- Fine-grained memory management and caching strategies to optimize memory usage especially for large library browsing. The average DRAM usage for browsing a library of **786 42MP RAW** files is around **767MB** while achieving smooth scrolling and instant preview generation.
- Written in modern C++20 with a focus on code quality, modularity, and maintainability (unfortunately, still largely a WIP).

### Professional Imaging Pipeline

- 32-bit floating-point processing pipeline.
- Support **ACES 2.0 Output Rendering**.
- Film-like highlight transition and sigmoid contrast curve.
- **CUBE** LUT support for creative color grading.
- Support JPEG/TIFF/PNG/EXR output with metadata write-back.
- Unlimited history stack with Git-like version control and branching.
- OpenImageIO/Exiv2-based image output with support for various formats and metadata handling.
- Planning to support HDR workflow and output in the future.

### Asset Management ("Sleeve" System)

- A simple but flexible inode-like file system using DuckDB as the storage backend, designed to manage both the original RAW files and the generated metadata (previews, thumbnails, edit history, etc.) in a unified way.
- Lean project management with a single project file that contains all the metadata and references to the original files, enabling easy project sharing and backup without worrying about missing sidecar files or broken links.
- Advanced search and filtering capabilities based on EXIF metadata. Planning to support semantic search and AI-assisted tagging in the future.

## System Requirements

- Windows 10/11 x64 for the current full CUDA/OpenGL editor build.
- macOS for an experimental CPU-only Qt application build. The editor backend is disabled there for now; future accelerated work on macOS is planned around Metal.
- NVIDIA GPU with CUDA support (minimum compute capability 6.0 (10-series or later), recommended 7.0+ (20-series or later) for optimal performance) and preferably 6GB+ VRAM for smooth performance with high resolution RAW files (40MP+).
- At least 8GB of system RAM (16GB+ recommended for larger libraries and smoother performance).
- 500MB of free disk space for the installation and temporary working files.
- 60+ MB for installation package and partial update support.

## Build from Source

This section mirrors the current setup in `CMakeLists.txt`, `pu-erh_lab/tests/CMakeLists.txt`, and NOTICE files.

### Windows (current full feature set)

### 1) Prerequisites

- Windows 10/11 x64
- Visual Studio 2022 (MSVC toolchain, x64)
- CMake 3.21+
- Ninja
- Git
- Qt 6 (MSVC 2022 x64), with `Widgets`, `Quick`, `OpenGL`, `OpenGLWidgets`, and `Test`
- NVIDIA CUDA Toolkit (optional, but recommended)

### 2) Dependency Layout Used by CMake

- Vendored header/source dependencies: `stduuid`, `uuid_v4`, `UTF8-CPP` (`utfcpp`), `nlohmann/json`, `MurmurHash3` (all these are required to install manually by the user, as they are not included in the repository)
- Package-managed dependencies (commonly resolved through vcpkg toolchain on Windows): `OpenCV`, `Eigen3`, `OpenGL`, `hwy`, `lcms2`, `OpenColorIO`, `OpenImageIO`, `libraw`, `xxHash`, `OpenMP`, `glib`
- Test framework: `googletest` (fetched with `FetchContent`)
- Windows local imported binaries: `DuckDB`, `Exiv2`, `easy_profiler`
- Lens correction dependency: the upstream `Lensfun` source checkout in `pu-erh_lab/src/third_party/lensfun`, built automatically by the top-level CMake build
- Additional Windows dependency for the bundled Lensfun build: `GLib2`. When the vcpkg toolchain is active, Pu-erh Lab will use `vcpkg/installed/<triplet>` automatically. `PUERHLAB_LENSFUN_GLIB2_BASE_DIR` is only needed to override that auto-detected location or to point at a non-vcpkg GLib2 package.

### 3) Initialize the Bundled Lensfun Source

Make sure the upstream Lensfun submodule is present before configuring Pu-erh Lab:

```powershell
git submodule update --init --recursive pu-erh_lab/src/third_party/lensfun
```

For details about the Windows GLib2 prerequisite used by the bundled build:

- [docs/lensfun_build/lensfun_local_build.md](docs/lensfun_build/lensfun_local_build.md)

### 4) Configure and Build Pu-erh Lab

Clone and initialize submodules:

```powershell
git clone --recursive https://github.com/zidage/PuerhLab.git
cd PuerhLab
```

Bootstrap local vcpkg if needed:

```powershell
.\vcpkg\bootstrap-vcpkg.bat
```

Recommended: use `cmd /c scripts\msvc_env.cmd ...` so MSVC environment variables are set automatically.
Adjust the Qt/easy_profiler paths below to your local environment.

```powershell
# Debug configure
cmd /c scripts\msvc_env.cmd --preset win_debug `
  -DCMAKE_PREFIX_PATH="D:/Qt/6.9.3/msvc2022_64/lib/cmake" `
  -Deasy_profiler_DIR="$PWD/pu-erh_lab/third_party/easy_profiler-v2.1.0-msvc15-win64/lib/cmake/easy_profiler"

# Debug build
cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 4
```

```powershell
# Release configure
cmd /c scripts\msvc_env.cmd --preset win_release `
  -DCMAKE_PREFIX_PATH="D:/Qt/6.9.3/msvc2022_64/lib/cmake" `
  -Deasy_profiler_DIR="$PWD/pu-erh_lab/third_party/easy_profiler-v2.1.0-msvc15-win64/lib/cmake/easy_profiler"

# Release build + install
cmd /c scripts\msvc_env.cmd --build --preset win_release --parallel 4
cmd /c scripts\msvc_env.cmd --install build/release --prefix build/install
```

If you are not using the vcpkg-provided `glib` package, or you want to override the auto-detected GLib2 location, add:

```powershell
-DPUERHLAB_LENSFUN_GLIB2_BASE_DIR="<path-to-glib2>"
```

Optional deploy tuning:

```powershell
# Include software OpenGL fallback DLL in the install package (larger package).
-DPUERHLAB_DEPLOY_SOFTWARE_OPENGL=ON

# Pass extra options to Qt deploy tooling (semicolon-separated list).
-DPUERHLAB_QT_DEPLOY_TOOL_OPTIONS="--no-compiler-runtime;--no-translations;--no-system-d3d-compiler;--no-system-dxc-compiler"
```

Create a ZIP package (CPack):

```powershell
cpack --config build/release/CPackConfig.cmake
```

### 5) Run Main/Demo Binaries

Common binaries after a Debug build:

```powershell
.\build\debug\pu-erh_lab\src\puerhlab_main.exe
.\build\debug\pu-erh_lab\tests\ImagePreview.exe
```

### 6) Tests and Dev Utilities

Current executable targets in `pu-erh_lab/tests/CMakeLists.txt`:

- `SampleTest`
- `SingleRawLoad`
- `SingleThumbnailLoad`
- `ColorTempCudaSanityTest`
- `SleeveFSTest`
- `ImportServiceTest`
- `SleeveServiceTest`
- `FilterServiceTest`
- `PipelineServiceTest`
- `EditHistoryMgmtServiceTest`
- `ThumbnailServiceTest`
- `ExportServiceTest`
- `AlbumBackendImportTest`
- `AlbumBackendProjectTest`
- `AlbumBackendFolderTest`
- `AlbumBackendImageDeleteTest`
- `CudaImageGeometryOpsTest` (only when CUDA is found)

CTest-discovered suites currently include `PipelineServiceTest`, `EditHistoryMgmtServiceTest`, and all `AlbumBackend*` targets:

```powershell
ctest --test-dir build/debug --output-on-failure
```

Standalone tests can be run directly as executables, for example:

```powershell
.\build\debug\pu-erh_lab\tests\SampleTest.exe
```

Some historical unit tests remain intentionally disabled/commented during refactoring.

Formatting/lint targets are available (clang-tidy integration is still partial):

```powershell
cmd /c scripts\msvc_env.cmd --build --preset win_debug --target format
cmd /c scripts\msvc_env.cmd --build --preset win_debug --target tidy
```

### macOS (experimental CPU-only app build)

Install the required dependencies with Homebrew:

```bash
brew install cmake ninja qt opencv opencolorio duckdb exiv2 glib libraw little-cms2 highway openimageio pkg-config xxhash eigen libomp
```

Configure and build the main Qt application:

```bash
git submodule update --init --recursive pu-erh_lab/src/third_party/lensfun
cmake --preset macos_debug
cmake --build --preset macos_debug --target puerhlab_main
```

Run the app:

```bash
./build/macos-debug/pu-erh_lab/src/puerhlab_main
```

Notes:

- The `macos_debug` and `macos_release` presets build the main app without CUDA, without the OpenGL editor, and without tests.
- If Homebrew is installed in a nonstandard prefix, pass `-DCMAKE_PREFIX_PATH=/path/to/prefix` when configuring.
- The current macOS build is meant to keep the Qt application runnable without CUDA/OpenGL. Opening the editor entry point will report that the editor backend is unavailable.
- Future macOS acceleration work is expected to target Metal rather than CUDA/OpenGL.

## Roadmap

Roadmap and ongoing milestones:

- [docs/roadmap/roadmap.md](docs/roadmap/roadmap.md)

## License

The `v0.1.1` tag and earlier releases remain under Apache-2.0.
Development after `v0.1.1` is licensed under `GPL-3.0-only`, with an additional permission under GPLv3 section 7 for combining/distributing required NVIDIA CUDA components.
See [LICENSE](LICENSE) and [NOTICE](NOTICE).
