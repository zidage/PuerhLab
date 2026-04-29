# Alcedo Studio

项目网站：[English](https://zidage.github.io/Alcedo/en/) | [简体中文](https://zidage.github.io/Alcedo/zh/)

<p align="right"><a href="./README.md">English</a> | <a href="./README.zh-CN.md"><strong>简体中文</strong></a></p>

![License](https://img.shields.io/badge/License-GPLv3-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900)
![C++](https://img.shields.io/badge/C++-20-blue)

Alcedo Studio 是一个开源 RAW 图像处理与数字资产管理（DAM）项目，旨在为摄影师提供一个轻量级、高性能、且在很大程度上兼容行业标准的照片编辑与库管理工作流新选择。

> Alcedo Studio _**不是**_ 现有商业软件或其他开源项目的替代品。

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
      <td><img src="docs/demo/welcome.png" alt="欢迎 / 项目加载" width="100%" /></td>
      <td>欢迎界面 —— 从统一的品牌入口加载或新建项目</td>
    </tr>
    <tr>
      <td><img src="docs/demo/album.png" alt="影集浏览" width="100%" /></td>
      <td>影集浏览 —— 文件夹树、响应式缩略图网格，右侧“图库概览”按拍摄日期 / 相机型号 / 镜头聚合</td>
    </tr>
    <tr>
      <td><img src="docs/demo/advance_color.png" alt="HSL 与 CDL 调色" width="100%" /></td>
      <td>HSL 与 CDL Lift / Gain 色轮，搭配实时 Waveform 波形监视器</td>
    </tr>
    <tr>
      <td><img src="docs/demo/drt.png" alt="色彩科学切换（ACES 2.0 / OpenDRT）" width="100%" /></td>
      <td>可切换色彩科学 —— ACES 2.0 或 OpenDRT，配合显示色彩空间、EOTF 与峰值亮度控制</td>
    </tr>
    <tr>
      <td><img src="docs/demo/lut.png" alt=".cube LUT 库" width="100%" /></td>
      <td>.cube LUT 库 —— 支持搜索、目录扫描与一键套用</td>
    </tr>
    <tr>
      <td><img src="docs/demo/history.png" alt="可分支的编辑历史" width="100%" /></td>
      <td>可分支的编辑历史 —— 支持撤销、折叠或从任意历史状态分支</td>
    </tr>
    <tr>
      <td><img src="docs/demo/output.png" alt="导出队列" width="100%" /></td>
      <td>导出队列 —— 批量导出，支持格式、位深、缩放与元数据选项</td>
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
- 简单精炼的项目文件，仅需一个 `.alcd` 文件即可保存整个项目的状态（包括库结构、每张照片的编辑历史和版本信息等），方便迁移和备份。
- 高级搜索和过滤功能，支持按文件名、拍摄日期、相机型号、镜头型号、曝光参数等多种条件组合搜索照片。
- 计划未来支持语义搜索（例如搜索“拍摄于2023年夏天，使用50mm镜头，光圈f/1.8，拍摄人像的照片”）。

## 系统要求

- Windows 10/11 x64：当前完整 CUDA/OpenGL 编辑器构建目标平台。
- macOS：当前提供面向 Apple 平台的 Metal 后端 Qt 主应用构建；现有 preset 会关闭传统 OpenGL 编辑器，但保留 Apple 原生图像处理后端。
- Windows/CUDA 构建建议使用支持 CUDA 的 NVIDIA GPU（最低计算能力 6.0，即 10 系列或更高；推荐 7.0+，即 20 系列或更高），并尽量配备 6GB+ VRAM 以流畅处理高分辨率 RAW 文件（40MP+）。
- macOS/Metal 构建需要支持 Metal 的 Mac 硬件。
- 至少 8GB 系统内存（建议 16GB+ 以获得更大的库和更流畅的性能）。
- 500MB 可用磁盘空间用于安装和临时工作文件。
- 60MB+ 用于安装包和部分更新支持。

## 源码构建

详细构建步骤（中英对照）已单独维护在：
- [docs/build_from_source.md](docs/build_from_source.md)

Detailed bilingual build instructions:
- [docs/build_from_source.md](docs/build_from_source.md)

快速命令：

```powershell
# 当前 CMake 布局所需子模块
git submodule update --init --recursive `
  alcedo_studio/src/third_party/lensfun `
  alcedo_studio/src/third_party/libultrahdr `
  alcedo_studio/src/third_party/metal-cpp

# Windows Debug（MSVC wrapper + preset）
cmd /c scripts\msvc_env.cmd --preset win_debug -DCMAKE_PREFIX_PATH="D:/Qt/6.9.3/msvc2022_64/lib/cmake"
cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 8

# Windows Release
cmd /c scripts\msvc_env.cmd --preset win_release -DCMAKE_PREFIX_PATH="D:/Qt/6.9.3/msvc2022_64/lib/cmake"
cmd /c scripts\msvc_env.cmd --build --preset win_release --parallel 8
cmd /c scripts\msvc_env.cmd --install build/release --prefix build/install

# macOS Debug 与打包
cmake --preset macos_debug
cmake --build --preset macos_debug --target alcedo_main
cmake --preset macos_release
cmake --build --preset macos_release
cmake --build --preset macos_package
```

## 路线图

开发里程碑见：

- [docs/roadmap/roadmap.md](docs/roadmap/roadmap.md)

## 许可证

`v0.1.1` tag 及之前的发布版本继续遵循 Apache-2.0。
`v0.1.1` 之后的开发版本遵循 `GPL-3.0-only`，并在根 `LICENSE` 中附带一个基于 GPLv3 第 7 节、用于组合/分发必需 NVIDIA CUDA 组件的补充许可。
详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

