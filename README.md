# Pu-erh Lab 🍵

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/License-Apache_2.0-blue) ![Stage](https://img.shields.io/badge/stage-pre--alpha-orange) ![C++](https://img.shields.io/badge/C%2B%2B-20-blue)

**Pu-erh Lab** is an open-source RAW photo processor and digital asset management (DAM) system, similar in vision to darktable and RawTherapee. It provides non-destructive RAW development, high-performance image processing, and a professional workflow engine built in modern C++.

> ⚠️ **Note:** This project is currently in the **very early stages of development (Pre-Alpha)**. Features are subject to change, and the codebase is under active construction.

## 📷 Early Demo
**Video Demo**: [BiliBili](https://www.bilibili.com/video/BV1bPcxzzEeM)
<table>
	<colgroup>
		<col style="width: 80%" />
		<col style="width: 20%" />
	</colgroup>
	<tbody>
		<tr>
			<td><img src="docs/demo/1.png" alt="Preview 1 (placeholder)" width="100%" /></td>
			<td>Modern UI Interface</td>
		</tr>
    <tr>
			<td><img src="docs/demo/4.png" alt="Preview 4 (placeholder)" width="100%" /></td>
			<td>Film-like Highlight Transition</td>
		</tr>
		<tr>
			<td><img src="docs/demo/2.png" alt="Preview 2 (placeholder)" width="100%" /></td>
			<td>LUT Support / Unlimited History Stack</td>
		</tr>
		<tr>
			<td><img src="docs/demo/3.png" alt="Preview 3 (placeholder)" width="100%" /></td>
			<td>Full-res Image Preview / Git-like Version Control</td>
		</tr>
	</tbody>
</table>


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




