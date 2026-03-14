# Changelog

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
