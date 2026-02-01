# Pu-erh Lab 🍵

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/License-Apache_2.0-blue) ![Stage](https://img.shields.io/badge/stage-pre--alpha-orange) ![C++](https://img.shields.io/badge/C%2B%2B-20-blue)

**Pu-erh Lab** is an open-source RAW photo processor and digital asset management (DAM) system, similar in vision to darktable and RawTherapee. It provides non-destructive RAW development, high-performance image processing, and a professional workflow engine built in modern C++.

> ⚠️ **Note:** This project is currently in the **very early stages of development (Pre-Alpha)**. Features are subject to change, and the codebase is under active construction.

## 📷 Early Demo Screenshot



### Image Rendering Previews 
Applied with 2383 LUT and highlight recovery.
<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/demo/preview_01.png" alt="Preview 01 (placeholder)" width="480" />
      <br />
    </td>
    <td width="50%" align="center">
      <img src="docs/demo/preview_02.png" alt="Preview 02 (placeholder)" width="480" />
      <br />
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/demo/preview_03.png" alt="Preview 03 (placeholder)" width="480" />
      <br />
    </td>
    <td width="50%" align="center">
      <img src="docs/demo/preview_04.png" alt="Preview 04 (placeholder)" width="480" />
      <br />
    </td>
  </tr>
</table>

<figure>
  <img src="docs/demo/showcase.gif" width="800">
  <figcaption>The custom UI design is still a work in progress. LUTs on display will not be shipped in the release due to copyright restrictions. </figcaption>
</figure>

## 🎯 Vision

Pu-erh Lab aims to provide a professional-grade workflow for photographers, combining robust asset management with a non-destructive, node-based editing pipeline. We focus on performance, usability, and extensibility.

## ✨ Key Technical Features 

### 🚀 High-Performance Core
- **Concurrency First:** Built on a tile-based rendering mechanism to maximize multi-core CPU utilization during image processing.
- **Modern C++:** Written in C++, trying to be **_blazingly fast_**!
- **GPU Acceleration:** Support GPU-accelerated processing using CUDA for real-time editing performance.

### 🎨 Professional Imaging Pipeline
- **GPU-accelerated Pipeline:** 32-bit pipeline with responsive previewing experience.
- **RAW Support:** GPU-accelerated **LibRaw** based decoding module.
- **Color Management:** ACES 2.0 support using OCIO and custom GPU implementation. LUT-based stylization, support 1D/3D LUTs in `.cube` format.
- **Non-Destructive Editing:** Flexible serializable architecture allows for infinite undo/redo and simple _version control_ without altering original files.

### 🗃️ Asset Management ("Sleeve" System)
- A custom abstraction layer designed specifically for handling massive photo libraries, providing efficient caching and path resolution.
- DuckDB-powered image database indexing with advanced seraching functionalities.

## 🚧 Development Status

We are currently working on the foundational architecture:

- [WIP] Basic RAW image decoding (LibRaw integration)
- [x] "Sleeve" filesystem abstraction layer
- [x] Basic pipeline model
- [x] GPU acceleration support
- [ ] User Interface (UI) implementation
- [ ] Non-destructive edit history serialization

The detailed development roadmap can be found [here](https://github.com/zidage/PuerhLab/blob/main/docs/roadmap/roadmap.md).

## 🔨 Building from Source

**Prerequisites:**
*   C++ Compiler supporting C++20 (MSVC, Clang, has not been tested with GCC yet)
*   CMake 3.20+
*   Git

**Build Steps:**
It is possible to build Pu-erh Lab right from the repository using CMake and vcpkg on Windows. But the dependencies list is still evolving, so please be patient.
```bash
# Clone the repository
git clone --recursive https://github.com/your-username/pu-erh_lab.git
cd pu-erh_lab

# Configure with CMake (vcpkg will bootstrap automatically)
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build --config Release
```



