# Pu-erh Lab

<p align="right"><a href="./README.md">English</a> | <a href="./README.zh-CN.md"><strong>简体中文</strong></a></p>

![License](https://img.shields.io/badge/License-Apache_2.0-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900)
![C++](https://img.shields.io/badge/C++-20-blue)

Pu-erh Lab (普洱工坊) 是一个开源 RAW 图像处理与数字资产管理（DAM）项目，旨在为摄影师提供一个轻量级、高性能、且在很大程度上兼容行业标准的照片编辑与库管理工作流新选择。

> Pu-erh Lab _**不是**_ 现有商业软件或其他开源项目的替代品。

## 早期演示

视频演示1：[BiliBili](https://www.bilibili.com/video/BV1bPcxzzEeM)

视频演示2 (带解说)：[BiliBili](https://www.bilibili.com/video/BV1sFfjBeE3n)

<table>
  <colgroup>
    <col style="width: 80%" />
    <col style="width: 20%" />
  </colgroup>
  <tbody>
    <tr>
      <td><img src="docs/demo/1.png" alt="现代化 UI 界面" width="100%" /></td>
      <td>现代UI界面</td>
    </tr>
    <tr>
      <td><img src="docs/demo/2.png" alt="高光过渡效果" width="100%" /></td>
      <td>仿胶片高光过渡效果</td>
    </tr>
    <tr>
      <td><img src="docs/demo/3.png" alt="LUT 与历史栈" width="100%" /></td>
      <td>LUT 支持 / 无限历史栈 / 类 Git 版本控制</td>
    </tr>
    <tr>
      <td><img src="docs/demo/4.png" alt="导出流程" width="100%" /></td>
      <td>高级导出功能</td>
    </tr>
  </tbody>
</table>

## 核心技术特性

### 高性能核心

- CUDA 加速的图像处理管线，有着当前业界领先的实时预览分辨率，能在现代 GPU 上以 60 FPS 流畅处理大尺寸 RAW 文件（例如 45MP）。
- 精细调整的内存管理与缓存策略，优化庞大影像库浏览时的内存使用，平均 DRAM 占用约 767MB（浏览包含 786 张 42MP RAW 文件的库）同时实现流畅滚动和即时预览生成。
- 使用现代 C++20 编写，注重代码质量、模块化和可维护性（估计是个长期老大难问题）。

### 专业图像处理流水线

- 32位浮点图像处理管线。
- 支持ACES 2.0 的”输出渲染（Output Rendering）“色彩管理体系。
- 负片般的高光过渡算法，适合人像和风光摄影（当然现在还没蒙版，没法让大伙画光）。
- 支持CUBE格式的LUT风格化调色，但需要是ACEScc->ACEScc的LUT。
- 支持带元数据写回的 JPEG/TIFF/PNG/EXR 输出。
- 无限的历史栈（^Z, ^Z, ^Z...），并且支持类似 Git 的版本控制（仅分支），可以随时回退到任意历史状态，或在不同版本之间切换对比。
- 基于OpenImageIO/Exiv2的影像元数据处理。
- 计划未来支持 HDR 工作流和输出（方便大家发小红书）。

### 资产管理（Sleeve 系统）

- 简单但灵活的inode式内置文件系统，基于数据库存储，支持文件夹和文件的层级结构。
- 简单精炼的项目文件，仅需一个 `.puerhproj` 文件即可保存整个项目的状态（包括库结构、每张照片的编辑历史和版本信息等），方便迁移和备份。
- 高级搜索和过滤功能，支持按文件名、拍摄日期、相机型号、镜头型号、曝光参数等多种条件组合搜索照片。
- 计划未来支持语义搜索（例如搜索“拍摄于2023年夏天，使用50mm镜头，光圈f/1.8，拍摄人像的照片”）。

## 系统要求

- Windows 10/11 x64（目前仅支持 Windows，未来可能支持 Linux/Mac）。
- NVIDIA GPU，支持 CUDA（最低计算能力 6.0（10 系列或更高），推荐 7.0+（20 系列或更高）以获得最佳性能），最好有 6GB+ VRAM 以流畅处理高分辨率 RAW 文件（40MP+）。
- 至少 8GB 系统内存（建议 16GB+ 以获得更大的库和更流畅的性能）。
- 500MB 可用磁盘空间用于安装和临时工作文件。
- 60MB+ 用于安装包和部分更新支持。

## 已知问题（v0.1.0）

### 影像处理管线

- 目前仅支持 **RGGB 拜尔阵列** 的 RAW 文件（有点惨）。支持的相机列表与 LibRaw 0.22.0 一致，但除去使用 X-Trans 传感器的部分富士机型和一些非 RGGB 模式的机型（例如松下 LUMIX S1R）。详细列表可以在 [LibRaw 文档](https://www.libraw.org/supported-cameras) 中找到。这也意味着 JPEG 和 TIFF 文件目前不支持导入和编辑。
- 对于一些较新的相机型号，由于当前 LibRaw 版本缺乏支持，图像可能会出现黑色边缘、白平衡不正确，甚至无法加载的问题。这预计将在未来集成更新的 LibRaw 版本后得到解决。在此情况下，即使图像可以正确处理，导出的文件可能仍然缺少 EXIF 元数据。
- 适配自 darktable 的“inpaint opposed”方法的高光重建算法在极端情况下（例如大面积过曝）可能会产生明显的伪影（artifacts）。未来计划提供一个“禁用”选项，并且可能会实现一个更牛的方法（估计得取决于未来 LLM 的能力了）。
- 版本控制目前还不支持分支，且在不同版本之间切换时可能会遇到预览中 LUT 没有正确应用的问题。可以通过重新应用 LUT 或等待未来的修复来解决这个问题。
- CDL 色轮可能对鼠标拖动有点过于敏感，一拖就变成RGB灯了。
- 在对比度调整为-100时，图像会变黑。
- 不支持用 Ctrl + Z 撤销，只能鼠标点那个Undo。
- 如果你编辑的是带有夕阳场景的照片，由于 ACES 2 对超出色域颜色的处理方式，天空的颜色可能开启来有点偏三文鱼色。可以通过使用 [aces-looks](https://github.com/priikone/aces-looks) 提供的 "Default_LMT" LUT 来缓解这个问题，该 LUT 模拟了 ACES 1.3 的外观，可以“修复”夕阳颜色问题。
  
### 资产管理

- 目前 UI 不支持从库中删除单张图片，也不支持在文件夹之间移动图片。底层 API 支持这些操作，但 UI 尚未实现。
- 如果导入了不受支持的 RAW 文件，可能会导致库损坏并在未来无法加载。因此请确保目前仅导入受支持的 RAW 文件。未来计划提供更健壮的处理方式。
- 由于我没有那么多 RAW 照片（比如说 10000+）来测试，库在那个规模下的可用性仍然未知。如果你有大量 RAW 文件并且愿意测试库的表现，请随时尝试并报告你遇到的任何问题。
- 缩略图生成可能会遇到一些数据竞争问题，导致生成的缩略图出现异常。但编辑会话和导出的图像不会受到这个问题的影响。
- 如果你使用的是**非英文** Windows 系统，并且没有启用系统设置中的“为全球语言支持使用 Unicode UTF-8”选项，可能会在遇到非 ASCII 文件路径时导致应用程序崩溃。如果由于兼容性问题无法启用 UTF-8 支持，请确保目前仅在文件路径中使用 ASCII 字符，包括安装路径、库路径、每个导入图像的路径和导出路径。

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
.\build\debug\pu-erh_lab\src\puerhlab_main.exe
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
