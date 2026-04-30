# Changelog

## [0.2.3] (21046ec..fd3f8f2) — 2026-04-08 ~ 2026-04-26

### Features
- **Project rebranded to Alcedo Studio**: Renamed the project from Puerh Lab to Alcedo Studio across the codebase, UI, and website, and added a new welcome screen. (`abdfa38`, `0ebb546`, `cc02941`)
- **WebGPU RAW processing backend**: Added experimental WebGPU support to the image buffer and introduced a full WebGPU-accelerated RAW decode pipeline — including RCD demosaic shaders, linear reference op, skeleton backend, and RCD demosaic performance tuning. (`3db42c6`, `8caa858`, `d09d5e5`, `4d7e041`, `f3cdde3`, `ab32232`)
- **Windows preview surface migrated to D3D12**: Ported the Windows preview surface from D3D11 to D3D12 in preparation for WebGPU support. (`1573e4a`)
- **Forward matrix support for RAW color**: Added forward matrix to the RAW color context and metadata extraction pipeline for improved color accuracy on supported cameras. (`0578f9d`)
- **DNG import and metadata improvements**: Optimized DNG file import performance, enhanced UI components, added DNG metadata extraction tests, and improved the DNG Converter recovery menu design. (`614bac2`, `ef33ff67`, `7481f37`)
- **Clarity operator improvements**: Improved Clarity operator quality and aligned its behavior between macOS and Windows. (`d7a79fe`, `efd30e4`)
- **Gesture operations on viewer**: Added pinch-to-zoom and pan gesture support to the image viewer. (`37f58f7`)
- **OCIO configuration enhancements**: Improved OCIO configuration handling and cross-platform path management. (`26096ac`)
- **LUT search and panel updates**: Added search support to the LUT selector and refreshed the LUT selection panel UI. (`5bb41c4`, `992bcd3`)
- **Inference backend migrated to ONNX**: Replaced the previous inference sidecar backend with ONNX Runtime. (`cf3a12e`)
- **i18n support for adjustment panels**: Added localization coverage for all adjustment panel strings. (`5e8920f`)
- **macOS installation script update**: Updated the macOS installation helper script. (`fd3f8f2`)

### UI
- **Comprehensive UI overhaul**: Overhauled the main theme and inspector panel, redesigned the tone, geometry, scope, versioning, and export panels, updated slider and folder styles, redesigned history cards, updated data display fonts, and added a pipeline profiler readout. (`4fb8ad7`, `7d4b345`, `5450a54`, `5f9153d`, `ba568cd`, `15e04a8`, `42599fa`, `3c1fbdb`, `4ccdc3d`, `6269a5c`, `9c3a10f`, `a96f7e7`, `3fa0b81`, `723e894`, `b87cbc8`, `aedd372`, `b273134`, `8e3297d`)

### Performance
- **VRAM optimization for large images**: Reduced peak VRAM consumption when processing 100MP+ images and capped the preview render resolution at 8K. (`1a9f09a`, `e49f5e6`)
- **Highlight reconstruction CUDA optimization**: Further tuned the CUDA highlight reconstruction kernel for improved throughput. (`4277677`)
- **Thumbnail and decode optimizations**: Optimized thumbnail downsample logic and general decode pipeline efficiency. (`4046e82`)

### Bug Fixes
- **RAW color matrix resolution fixes**: Fixed CCM resolution errors for DNG files and general camera matrix matching. (`83370a8`, `060d887`)
- **D3D12 preview crash**: Fixed a crash-to-desktop when initializing the D3D12 preview surface. (`11af467`)
- **Lens correction crop**: Fixed broken crop output after lens correction is applied. (`66d6fbb`)
- **Curve control behavior and rendering**: Fixed curve control interaction behavior and panel corner rendering. (`767bcf9`, `b591a56`)
- **Editor font rendering on Windows**: Fixed incorrect font rendering in the editor on Windows. (`919fc88`)
- **Miscellaneous UI fixes**: Fixed incorrect collapse/expand button color and inconsistent panel headline design. (`17fe748`, `b97e016`)

## [0.2.2] (6def338..17363e4) — 2026-03-22 ~ 2026-04-08

### Features
- **Nikon HE / HE* RAW recovery workflow**: Added Nikon HE-compressed NEF detection during import, a guided Adobe DNG Converter recovery dialog, automatic project cleanup/reimport after conversion, and macOS support for the same flow. Linear DNG inputs are now accepted, so converted files can go straight back into the RAW pipeline. (`b8e4962`, `dc86707`, `d32992d`, `0f85b8a`)
- **Highlight reconstruction and tone refactor**: Reworked RAW highlight recovery on CUDA and Metal into a multi-pass clipped-mask/chrominance-accumulation pipeline, and refactored Highlights/Shadows adjustments around a shared tone curve with new tests for knee behavior and chroma preservation. (`352d3d2`, `a4218c5`, `478205b`, `624cc24`)
- **LUT browser & Look panel redesign**: Rebuilt the editor side-panel layout, split out a dedicated Look panel, and added a LUT catalog/browser with `.cube` header validation, missing/invalid state display, quick folder open/refresh actions, and better selection persistence. (`955b47d`, `b8e4962`, `83583f0`)
- **Color, export, and metadata upgrades**: Added ICC profile embedding on Windows, expanded built-in export profile support, added EXIF details/source-path UI, added macOS scopes, and improved camera metadata resolution for tricky bodies such as Hasselblad. (`6def338`, `93c0b08`, `1f36cd6`, `da0102d`, `912dc2d`)
- **Experimental PuerhMind additions**: Added the Rust-based semantic/inference sidecar, CLIP text/vision services, simple image labeling, and a macOS inference demo path. (`73ad1c4`, `48a794c`, `a5ed1f7`, `edac38d`)

### Performance
- **High-resolution RAW decode acceleration**: Split the CUDA RAW path into dedicated full-frame and tiled execution modes, added active-area-aware crop handling, and reduced peak cost for very large Bayer files. (`624cc24`, `2d80f39`)
- **Less GPU copying and redundant work**: Added GPU buffer sharing and no-op detection for geometry stages so resize/crop passes can skip redundant work or avoid extra copies when possible. (`624cc24`, `2d80f39`)
- **Kernel fusion and intermediate reuse**: Combined highlight correction with RGBA packing, introduced reusable CUDA/Metal workspaces, tightened several low-level RAW kernels, and improved thumbnail / inference-side throughput. (`624cc24`, `2d80f39`, `22bb73b`)

### Bug Fixes
- **Tone and color stability**: Fixed the contrast `-100` all-black issue, corrected color temperature UI refresh behavior, and improved camera matrix matching for Hasselblad files. (`6294602`, `83583f0`, `912dc2d`)
- **Workflow and platform stability**: Fixed export dialog layout/parameter issues, added source-missing notifications in the album UI, and added CUDA driver version requirement probing on startup. (`9ad8384`, `d3083ff`, `4c21e10`, `17363e4`)

## [0.2.1] (044f948..6d0ff5b) — 2026-03-20 ~ 2026-03-20

### Features
- **CUDA support for X-Trans RAW**: Extended the GPU RAW decode path to Fuji X-Trans sources instead of limiting CUDA acceleration to classic Bayer cameras. (`044f948`)

## [0.2.0] (03344c0..b8c2fa3) — 2026-03-07 ~ 2026-03-14

### Features
- **Cross-platform rendering expansion**: Added macOS build support and integrated the Metal pipeline (briefly: raw/resize/lens utilities, pipeline wiring, and performance/refactor passes) (`5eed41d`, `0a37cfa`, `aefa6f0`)
- **macOS visual pipeline upgrades**: Added basic color management and experimental HDR support on macOS (`880234c`, `4c879c3`)
- **Windows preview backend update**: Ported Windows preview surface to D3D11 (`3a079ad`)
- **Internationalization**: Added i18n support, language selection UI adjustments, and zh-CN font updates (`2caeaed`, `9657bf5`, `44b5401`)
- **New scopes & controls**: Added histogram/waveform display, aspect ratio selection, thumbnail waiting animation, and reset adjustments support (`5f47c71`, `e559e1e`, `85a4440`, `2c18f7f`)
- **Versioning UI refresh**: Improved versioning UI design (`a0a5931`)

### Bug Fixes
- **Windows build stability**: Fixed multiple Windows compile issues during cross-platform integration (`e794c4e`, `a6d8968`, `2eca003`, `4f89c41`)
- **Metal pipeline path fix**: Corrected wrong geometry pipeline path in Metal (`34242aa`)
- **Renderer include/path fixes**: Updated include path handling for OpenGL viewer renderer (`cefe155`)
- **Editor background issue**: Fixed editor background issue in reset-adjustments workflow (`2c18f7f`)

### Documentation
- Added changelog documentation (`805996f`)
- Added demo website and updated project website content (`f6b76d8`, `2eff447`)
- Updated README content (performance data, removed outdated video link) (`1699e67`, `adc1912`)

### Miscellaneous
- Added website deployment GitHub Actions workflow (`385ecec`)
- Added/updated dependency submodules (`metal-cpp`, `libultrahdr`) and Windows support integration (`7ffd7c5`, `65e1372`)
- Removed unnecessary `third_party` folder cleanup (`b8c2fa3`)

## [0.1.2] (846e9d3..03344c0) — 2026-03-01 ~ 2026-03-07

### Features
- **OpenDRT support**: Added support for OpenDRT (Open Display Rendering Transform), licensed under GPLv3 (`8c9e62a`)
- **Rendering transform selection**: Added support for selecting different rendering transforms (RT) in the pipeline (`6d94167`)
- **Image deletion**: Added support for deleting images from the project (`197df08`)
- **Filter UI improvements**: Improved the filter UI for better usability (`874c93b`)
- **Project font change**: Changed the font used in the project UI (`6c4c6ad`)

### Bug Fixes
- **CCT/Tint resolution**: Fixed color correlated temperature (CCT) and tint resolution calculation (`2d1efc9`)
- **File name display**: Fixed file name display issues in the editor and exporter (`20fe29b`)
- **Raw processing race conditions**: Fixed race conditions during raw image processing (`9dd3e42`)
- **Color management resolution**: Fixed name normalization error in color management resolution (`665b442`)

### Documentation
- Updated README with lensfun installation details (`537670d`)
- Updated core libraries listing in README (`c691484`)
- Updated lensfun build documentation (`3a40dd0`)
- Updated source dependencies information (`c96a980`)
- General README updates (`6a233e7`)

### Miscellaneous
- Updated LICENSE back to GPLv3 (`03344c0`)

---

# 更新日志

## [0.2.3] (21046ec..fd3f8f2) — 2026-04-08 ~ 2026-04-26

### 新功能
- **项目更名为 Alcedo Studio**：将项目从 Puerh Lab 更名为 Alcedo Studio，涵盖代码库、UI 及网站，并新增欢迎页面。(`abdfa38`, `0ebb546`, `cc02941`)
- **WebGPU RAW 处理后端**：新增实验性 WebGPU 支持，为图像缓冲区引入完整的 WebGPU 加速 RAW 解码管线，包括 RCD 去马赛克着色器、线性参考算子、骨架后端及 RCD 去马赛克性能优化。(`3db42c6`, `8caa858`, `d09d5e5`, `4d7e041`, `f3cdde3`, `ab32232`)
- **Windows 预览曲面迁移至 D3D12**：将 Windows 预览 Surface 从 D3D11 迁移到 D3D12，为 WebGPU 支持做准备。(`1573e4a`)
- **RAW 色彩正向矩阵支持**：在 RAW 色彩上下文和元数据提取管线中加入正向矩阵，提升支持机型的色彩精准度。(`0578f9d`)
- **DNG 导入与元数据改进**：优化 DNG 文件导入性能，增强 UI 组件，新增 DNG 元数据提取测试，并改进 DNG Converter 恢复菜单设计。(`614bac2`, `ef33ff67`, `7481f37`)
- **清晰度算子改进**：提升清晰度算子质量，并统一 macOS 与 Windows 平台的行为表现。(`d7a79fe`, `efd30e4`)
- **预览区手势操作**：为图像预览区添加捏合缩放与平移手势支持。(`37f58f7`)
- **OCIO 配置增强**：改进 OCIO 配置处理方式与跨平台路径管理。(`26096ac`)
- **LUT 搜索与面板更新**：为 LUT 选择器添加搜索支持，并刷新 LUT 面板 UI。(`5bb41c4`, `992bcd3`)
- **推理后端迁移至 ONNX**：将推理 sidecar 后端替换为 ONNX Runtime。(`cf3a12e`)
- **调整面板 i18n 支持**：为所有调整面板字符串补全本地化支持。(`5e8920f`)
- **macOS 安装脚本更新**：更新 macOS 安装辅助脚本。(`fd3f8f2`)

### 界面
- **全面 UI 大改版**：重做主题与检查器面板，重新设计色调、几何、波形、版本控制及导出面板，更新滑块与文件夹样式，重设历史记录卡片，更新数据显示字体，并添加管线性能分析输出。(`4fb8ad7`, `7d4b345`, `5450a54`, `5f9153d`, `ba568cd`, `15e04a8`, `42599fa`, `3c1fbdb`, `4ccdc3d`, `6269a5c`, `9c3a10f`, `a96f7e7`, `3fa0b81`, `723e894`, `b87cbc8`, `aedd372`, `b273134`, `8e3297d`)

### 性能优化
- **大尺寸图像 VRAM 优化**：降低 100MP 以上图像处理时的峰值 VRAM 消耗，并将预览渲染分辨率上限设定为 8K。(`1a9f09a`, `e49f5e6`)
- **高光恢复 CUDA 优化**：进一步调优 CUDA 高光恢复内核，提升处理吞吐量。(`4277677`)
- **缩略图与解码优化**：优化缩略图降采样逻辑及整体解码管线效率。(`4046e82`)

### 缺陷修复
- **RAW 色彩矩阵解析修复**：修复 DNG 文件的 CCM 解析错误及通用相机矩阵匹配问题。(`83370a8`, `060d887`)
- **D3D12 预览崩溃**：修复 D3D12 预览 Surface 初始化时的崩溃问题。(`11af467`)
- **镜头校正裁切**：修复应用镜头校正后裁切输出异常的问题。(`66d6fbb`)
- **曲线控制行为与渲染**：修复曲线控制交互行为和面板圆角渲染问题。(`767bcf9`, `b591a56`)
- **Windows 编辑器字体渲染**：修复 Windows 上编辑器中字体渲染错误。(`919fc88`)
- **其他 UI 修复**：修复收起/展开按钮颜色错误及面板标题设计不一致问题。(`17fe748`, `b97e016`)

## [0.2.2] (6def338..17363e4) — 2026-03-22 ~ 2026-04-08

### 新功能
- **Nikon HE / HE* RAW 恢复流程**：在导入阶段新增 Nikon HE 压缩 NEF 检测，加入引导式 Adobe DNG Converter 恢复对话框，支持转换后自动清理项目占位并重新导入；同时补齐 macOS 对同一流程的支持。线性 DNG 现在也可直接重新进入 RAW 管线。 (`b8e4962`, `dc86707`, `d32992d`, `0f85b8a`)
- **高光恢复与明暗部算法重构**：将 CUDA / Metal 上的 RAW 高光恢复重写为“裁剪掩码 + 色度统计 + 重建”的多阶段流程，并把 Highlights / Shadows 调整重构为共享色调曲线，补充了针对 knee 行为与色彩保持的测试。 (`352d3d2`, `a4218c5`, `478205b`, `624cc24`)
- **LUT 浏览器与 Look 面板大改版**：重做编辑器侧边栏结构，拆出独立 Look 面板，并新增 LUT 目录浏览器，支持 `.cube` 头信息校验、缺失/损坏状态提示、快速打开文件夹/刷新，以及更稳定的当前选择保持。 (`955b47d`, `b8e4962`, `83583f0`)
- **色彩、导出与元数据链路升级**：新增 Windows ICC profile 嵌入、补充内置导出 profile、加入 EXIF 详情/源路径显示、补齐 macOS scopes，并改进 Hasselblad 等机型的元数据解析。 (`6def338`, `93c0b08`, `1f36cd6`, `da0102d`, `912dc2d`)
- **实验性 PuerhMind 能力接入**：加入 Rust 语义/推理 sidecar、CLIP 文本与视觉服务、基础图片标注能力，以及 macOS 推理 demo。 (`73ad1c4`, `48a794c`, `a5ed1f7`, `edac38d`)

### 性能优化
- **高分辨率 RAW 解码提速**：将 CUDA RAW 路径拆分为 full-frame 与 tiled 两种执行模式，引入基于 active area 的裁切处理，显著降低超大尺寸 Bayer 文件的峰值开销。 (`624cc24`, `2d80f39`)
- **减少 GPU 拷贝与空操作**：为几何阶段加入 GPU buffer 共享与 no-op 检测，让 resize / crop 在可跳过时直接跳过，在可复用时避免额外拷贝。 (`624cc24`, `2d80f39`)
- **融合内核与中间结果复用**：将高光校正与 RGBA 打包合并执行，引入可复用的 CUDA / Metal workspace，并同步优化多处底层 RAW kernel、缩略图链路与推理侧吞吐。 (`624cc24`, `2d80f39`, `22bb73b`)

### 缺陷修复
- **明暗部与色彩稳定性**：修复 Contrast 为 `-100` 时整张图变黑的问题，修正色温 UI 刷新异常，并改进 Hasselblad 文件的相机矩阵匹配。 (`6294602`, `83583f0`, `912dc2d`)
- **工作流与平台稳定性**：修复导出对话框布局与参数交互问题，补充相册中源文件缺失提示，并在启动时增加 CUDA 驱动版本要求探测。 (`9ad8384`, `d3083ff`, `4c21e10`, `17363e4`)

## [0.2.1] (044f948..6d0ff5b) — 2026-03-20 ~ 2026-03-20

### 新功能
- **CUDA 支持 X-Trans RAW**：将 GPU RAW 解码能力从经典 Bayer 机型扩展到 Fuji X-Trans 源文件。 (`044f948`)

## [0.2.0] (03344c0..b8c2fa3) — 2026-03-07 ~ 2026-03-14

### 新功能
- **跨平台渲染扩展**：新增 macOS 编译支持并完成 Metal 流水线集成（简述：Raw/缩放/镜头校正能力接入、流水线贯通，以及性能优化与重构） (`5eed41d`, `0a37cfa`, `aefa6f0`)
- **macOS 视觉流水线升级**：新增 macOS 基础色彩管理与实验性 HDR 支持 (`880234c`, `4c879c3`)
- **Windows 预览后端更新**：将 Windows 预览 Surface 移植到 D3D11 (`3a079ad`)
- **国际化支持**：新增 i18n、优化语言选择 UI，并更新 zh-CN 字体 (`2caeaed`, `9657bf5`, `44b5401`)
- **新示波与控制能力**：新增直方图/波形显示、画幅比例选择、缩略图等待动画，以及重置调整支持 (`5f47c71`, `e559e1e`, `85a4440`, `2c18f7f`)
- **版本信息界面优化**：改进 versioning UI 设计 (`a0a5931`)
- **改动重置支持**：支持用户通过双击滑块来重置调整参数。

### 缺陷修复
- **Windows 构建稳定性**：修复跨平台集成过程中多处 Windows 编译问题 (`e794c4e`, `a6d8968`, `2eca003`, `4f89c41`)
- **Metal 流水线路径修复**：修复 Metal 中几何管线路径错误 (`34242aa`)
- **渲染器包含路径修复**：修复 OpenGL viewer renderer 的 include/path 处理 (`cefe155`)
- **编辑器背景问题**：修复重置调整流程中的编辑器背景问题 (`2c18f7f`)

### 文档更新
- 新增 changelog 文档 (`805996f`)
- 新增 demo 网站并更新项目网站内容 (`f6b76d8`, `2eff447`)
- 更新 README（性能数据、移除过时视频链接） (`1699e67`, `adc1912`)

### 其他
- 新增网站部署 GitHub Actions 工作流 (`385ecec`)
- 新增/更新依赖子模块（`metal-cpp`、`libultrahdr`）并集成 Windows 支持 (`7ffd7c5`, `65e1372`)
- 清理并移除不再需要的 `third_party` 目录 (`b8c2fa3`)

## [0.1.2] (846e9d3..03344c0) — 2026-03-01 ~ 2026-03-07

### 新功能
- **OpenDRT 支持**：新增 OpenDRT（开放显示渲染变换）支持，采用 GPLv3 许可证 (`8c9e62a`)
- **渲染变换选择**：支持在流水线中选择不同的渲染变换（RT） (`6d94167`)
- **图片删除**：新增从项目中删除图片的功能 (`197df08`)
- **筛选器 UI 改进**：优化筛选器界面，提升可用性 (`874c93b`)
- **项目字体更换**：更换了项目 UI 使用的字体 (`6c4c6ad`)

### 缺陷修复
- **CCT/Tint 分辨率**：修复了色温（CCT）和色调（Tint）分辨率的计算问题 (`2d1efc9`)
- **文件名显示**：修复了编辑器和导出器中文件名显示异常的问题 (`20fe29b`)
- **Raw 处理竞态条件**：修复了 Raw 图像处理过程中的竞态条件 (`9dd3e42`)
- **色彩管理分辨率**：修复了色彩管理分辨率中名称归一化错误 (`665b442`)

### 文档更新
- 更新 README，添加 lensfun 安装说明 (`537670d`)
- 更新 README 中的核心库列表 (`c691484`)
- 更新 lensfun 构建文档 (`3a40dd0`)
- 更新源码依赖信息 (`c96a980`)
- 常规 README 更新 (`6a233e7`)

### 其他
- 将许可证恢复为 GPLv3 (`03344c0`)
