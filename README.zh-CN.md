# Pu-erh Lab

<p align="right"><a href="./README.md">English</a> | <a href="./README.zh-CN.md"><strong>简体中文</strong></a></p>

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/License-Apache_2.0-blue)
![Stage](https://img.shields.io/badge/stage-pre--alpha-orange)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue)

Pu-erh Lab 是一个开源 RAW 图像处理与数字资产管理（DAM）项目。注重高性能、非破坏式图像流水线，以及面向相册/项目工作流的服务化架构。

> 项目阶段：**Pre-Alpha**。接口与行为仍在快速迭代中，后续可能继续调整。

## 早期演示

视频演示：[BiliBili](https://www.bilibili.com/video/BV1bPcxzzEeM)

<table>
  <colgroup>
    <col style="width: 80%" />
    <col style="width: 20%" />
  </colgroup>
  <tbody>
    <tr>
      <td><img src="docs/demo/1.png" alt="现代化 UI 界面" width="100%" /></td>
      <td>Modern UI Interface</td>
    </tr>
    <tr>
      <td><img src="docs/demo/2.png" alt="高光过渡效果" width="100%" /></td>
      <td>Film-like Highlight Transition</td>
    </tr>
    <tr>
      <td><img src="docs/demo/3.png" alt="LUT 与历史栈" width="100%" /></td>
      <td>LUT Support / Unlimited History Stack / Git-like Version Control</td>
    </tr>
    <tr>
      <td><img src="docs/demo/4.png" alt="导出流程" width="100%" /></td>
      <td>Advanced Exporting Feature</td>
    </tr>
  </tbody>
</table>

## 愿景

Pu-erh Lab 目标是为摄影工作流提供一套专业方案，核心是把以下能力统一起来：

- 面向大规模素材库的稳定数字资产管理，
- 分阶段、非破坏式的图像编辑流水线，
- 面向反复调图场景的高响应交互体验。

## 核心技术特性

### 高性能核心

- 并发优先的架构设计，结合 tile 渲染与任务调度。
- 基于现代 C++20 的模块化代码组织。
- 可选 CUDA 加速路径，用于 RAW 处理与部分流水线阶段。

### 专业图像处理流水线

- 以 32-bit float 为核心的数据处理链路（预览与导出）。
- 基于 LibRaw 的 RAW 解码，并支持 CPU/GPU 算子分离实现。
- 使用 OpenColorIO + lcms2 的色彩管理体系，支持 `.cube` LUT。
- 支持非破坏式编辑与历史/版本化管理。

### 资产管理（Sleeve 系统）

- 自定义 Sleeve 抽象，用于文件夹/文件组织与大库导航。
- 基于 DuckDB 的元数据与索引存储（mapper/service 分层）。
- 覆盖导入、过滤、缩略图、流水线管理、导出的应用服务层。

## 近期进展

近几轮迭代中，主要更新包括：

- 几何编辑流程：交互式裁剪/旋转，以及 `CROP_ROTATE` 算子接入编辑流水线。
- 色调与色彩流程：HLS 参数模型扩展、曲线编辑体验增强。
- 色彩输出链路：ODT/LUT 资源缓存与生命周期管理优化。
- 预览与显存：预览质量/速度优化，以及 VRAM 占用优化。
- DAM/应用层：项目与文件夹管理增强，持续向服务层架构迁移。
- 历史与版本：历史栈能力完善、状态同步修复、相册重载流程改进。
- 导出流程：导出进度反馈与稳定性修复。

## 源码构建（仅 Windows）

本节仅覆盖 Windows，内容对齐当前 `CMakeLists.txt` 与 `CMakePresets.json`。

### 1）环境要求

- Windows 10/11 x64
- Visual Studio 2022（MSVC x64 工具链）
- CMake 3.21+
- Ninja
- Qt 6（MSVC 2022 x64 版本）
- Git
- 可选：NVIDIA CUDA Toolkit（检测到后自动启用 CUDA 相关目标）

### 2）CMake 依赖解析

当前顶层 CMake 主要依赖/查找：

- Qt6：`Widgets`, `Quick`, `OpenGL`, `OpenGLWidgets`（QML Demo 还需要 `Qml`, `QuickControls2`, `QuickDialogs2`, `QuickEffects`）
- 核心库：`OpenCV`, `Eigen3`, `OpenGL`, `glad`, `hwy`, `lcms2`, `OpenColorIO`, `OpenImageIO`, `libraw`, `xxHash`
- 并行与测试：`OpenMP`, `googletest`（通过 `FetchContent` 自动拉取）
- Windows 分析器：`easy_profiler`
- 本地第三方二进制：DuckDB 来自 `pu-erh_lab/third_party/libduckdb-windows`（Exiv2 也配置了本地回退路径）

### 3）配置与编译

先拉取源码与子模块：

```powershell
git clone --recursive https://github.com/zidage/PuerhLab.git
cd PuerhLab
```

如需初始化本地 vcpkg：

```powershell
.\vcpkg\bootstrap-vcpkg.bat
```

推荐使用 `scripts/msvc_env.cmd`，自动注入 MSVC 构建环境：
下面命令中的 Qt/easy_profiler 路径请按本机实际目录调整。

```powershell
# Debug 配置
.\scripts\msvc_env.cmd --preset win_debug `
  -DCMAKE_PREFIX_PATH="D:/Qt/6.9.3/msvc2022_64/lib/cmake" `
  -Deasy_profiler_DIR="$PWD/pu-erh_lab/third_party/easy_profiler-v2.1.0-msvc15-win64/lib/cmake/easy_profiler"

# Debug 编译
.\scripts\msvc_env.cmd --build build/debug --parallel
```

```powershell
# Release 配置
.\scripts\msvc_env.cmd --preset win_release `
  -DCMAKE_PREFIX_PATH="D:/Qt/6.9.3/msvc2022_64/lib/cmake" `
  -Deasy_profiler_DIR="$PWD/pu-erh_lab/third_party/easy_profiler-v2.1.0-msvc15-win64/lib/cmake/easy_profiler"

# Release 编译 + 安装
.\scripts\msvc_env.cmd --build build/release --parallel
.\scripts\msvc_env.cmd --install build/release --prefix build/install
```

可选部署参数：

```powershell
# 在安装包中包含软件 OpenGL 回退 DLL（体积更大）。
-DPUERHLAB_DEPLOY_SOFTWARE_OPENGL=ON

# 传递额外参数给 Qt 部署工具（使用分号分隔）。
-DPUERHLAB_QT_DEPLOY_TOOL_OPTIONS="--no-compiler-runtime;--no-translations;--no-system-d3d-compiler;--no-system-dxc-compiler"
```

打包 ZIP（CPack）：

```powershell
cpack --config build/release/CPackConfig.cmake
```

### 4）运行 Demo

Debug 构建后常用可执行文件路径：

```powershell
.\build\debug\pu-erh_lab\tests\CompleteUIDemo.exe
.\build\debug\pu-erh_lab\tests\ThumbnailAlbumQtDemo.exe
.\build\debug\pu-erh_lab\tests\ImagePreview.exe
.\build\debug\pu-erh_lab\tests\gui_pocs\album_editor_qml\AlbumEditorQmlDemo.exe
```

### 5）测试与开发工具

- 当前不少测试以独立可执行文件形式运行，位于 `build\debug\pu-erh_lab\tests\`。
- 在应用层重构期间，部分历史单测在 `pu-erh_lab/tests/CMakeLists.txt` 中暂时关闭。
- 可用格式化/静态检查目标：

```powershell
.\scripts\msvc_env.cmd --build build/debug --target format
.\scripts\msvc_env.cmd --build build/debug --target tidy
```

## 路线图

开发里程碑见：

- [docs/roadmap/roadmap.md](docs/roadmap/roadmap.md)

## 许可证

Apache-2.0。详见 [LICENSE](LICENSE)。
