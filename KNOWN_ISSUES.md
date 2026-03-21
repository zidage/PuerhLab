# Known Issues


## v0.2.0

### Common Issues

#### Pipeline

- Only support **Bayer** RAW images with RGGB pattern. Support for other patterns will be added in future releases.
- Nikon's High Efficiency Compressed RAW format is currently not supported. You can convert these files to DNG format to use them in the app.
- Lens correction will fail if the focal length data is not properly read by the metadata parser. This issue frequently occurs on Canon mirrorless cameras.
- Color Temp/Tint adjustments will fail if the camera model is not properly read by the metadata parser. This issue may occur on some mobile devices.
- ACES 2.0 does not support HDR processing, where the peak luminance option is currently broken.
- Histogram/Waveform display may not be rendered when switching between two scopes. You can workaround this issue by dragging a slider in the control panel to trigger a re-render.

#### Versioning

- Version switching may cause LUT to be applied incorrectly. You can solve this by re-selecting the LUT in the control panel.
- The versioning UI is too compact to display long version names.
- The "Plain" version does not reset adjustments to default values at all. Use "Incremental" version instead.

#### Assets Management

- If you have imported unsupported images, the thumbnail will not be generated. You can delete this image from the project and re-import it after the issue is fixed in future releases.
- If you have delete an image from your **disk**, the current implementation will not be able to relink the image, and all adjustments will be lost.
- The project file is not backward compatible. Please do not open a project created in v0.1.2 or earlier versions, as it may behave unexpectedly.
  
#### Other Issues

- Some texts miss localization and may appear in English even if you have set the app to another language. 

### Windows Issues

- The HDR preview is not available on Windows due to a technical limitation in the current implementation. But you can still export Ultra HDR images when you set the DRT to P3/ST2084 or Rec.2020/ST2084.
- Does not support Color Management System (CMS) on Windows. If you use a wide-gamut monitor, limit your monitor to sRGB. Otherwise, the output colors may be inaccurate as the output image will not be  properly tagged with the correct color space. This issue will be fixed in future releases.
- _Please update your GPU drivers to the latest version_. The mimimum required GPU driver version is 570.xx. Older drivers will cause crashes.

### macOS Issues

- I only have access to a MacBook Air M4 running macOS Tahoe (macOS 26), so I can only test the app on this specific hardware and software configuration. The app is **built for Apple Silicon**. If you encounter any issues on other ARM Mac models or macOS versions that support Metal, please report them to me and I will try to fix them in future releases.
- Since I haven't enrolled in the Apple Developer Program, the app is **NOT signed and notarized**. It will be blocked by macOS's safety measures. Workaround _may_ exist but will not be discussed here. It is recommended to build the app from source code if you are concerned about security.
- The HDR preview and color management **IS** available on macOS, and the export images will be tagged with the color space you choose in the DRT settings. However, the UI color will be inaccurate if you are using a color space other than sRGB, as the app does not support color management in the UI.
- It is recommended to have 16GB of RAM for the app to run smoothly. The app may perform poorly or even crash on machines with less RAM, especially when processing large images. 

# 已知问题

## v0.2.0

### 常见问题

#### 编辑流程

- 目前仅支持 RGGB 排列的 Bayer RAW 图像，其他排列的支持将会在未来版本中添加。
- 尼康高效率压缩 RAW 格式目前不受支持。你可以将这些文件转换为 DNG 格式以在应用中使用。
- 如果元数据解析器未正确读取焦距数据，镜头校正将会失败。这个问题经常发生在佳能无反相机上。
- 如果元数据解析器未正确读取相机型号，色温/色调调整将会失败。这个问题可能会发生在一些移动设备上。
- ACES 2.0 不支持 HDR 处理，目前峰值亮度选项是无效的。
- 切换两个波形图时，直方图/波形显示可能无法正确渲染。你可以通过在下方面板中拖动一个滑块来触发重新渲染以解决这个问题。
  
#### 版本控制

- 切换版本可能会导致 LUT 应用不正确。你可以通过在控制面板中重新选择 LUT 来解决这个问题。
- 版本控制界面过于紧凑，无法显示较长的版本名称。
- "Plain" 版本根本不会将调整重置为默认值。请改用 "Incremental" 版本。

#### 资源管理

- 如果你导入了不受支持的图像（例如，JPEG, TIFF，尼康高效率压缩RAW），将无法生成缩略图。你可以从项目中删除这个图像，并在未来版本修复这个问题后重新导入它。
- 如果你从磁盘中删除了一个图像，当前的实现将无法重新链接这个图像，并且所有调整都将丢失。
- 项目文件不向后兼容。请不要打开在 v0.1.2 或更早版本中创建的项目，因为它可能会表现异常。
  
#### 其他问题

- 一些文本缺乏中文翻译，即使你将应用设置为其他语言，它们也可能以英文显示。
  
### Windows 问题

- 由于当前实现的技术限制，HDR 预览在 Windows 上不可用。但当你将 DRT 设置为 P3/ST2084 或 Rec.2020/ST2084 时，你仍然可以导出 Ultra HDR 图像。
- Windows 上不支持颜色管理系统（CMS）。如果你使用宽色域显示器，请将显示器限制在 sRGB。否则，预览颜色可能不准确，并且输出图像将不会正确标记为正确的色彩空间。这个问题将在未来版本中修复。
- _请将你的 GPU 驱动程序更新到最新版本_。最低要求的 GPU 驱动程序版本是 570.xx。较旧的驱动程序将导致崩溃。

### macOS 问题

- 我只有一台运行 macOS Tahoe (macOS 26) 的 MacBook Air M4，因此我只能在这个特定的硬件和软件配置上测试应用。该应用**为 Apple Silicon 构建**。如果你在其他支持 Metal 的 ARM Mac 型号或 macOS 版本上遇到任何问题，请向我报告。
- 由于我没有加入 Apple Developer Program，应用**未签名和未公证**。它将被 macOS 的安全措施阻止启动。可能存在解决方法，但这里不会讨论。如果你担心安全问题，建议从源代码构建应用。
- HDR 预览和颜色管理在 macOS 上可用，并且导出图像将被标记为你在 DRT 设置中选择的色彩空间。然而，如果你使用非 sRGB 的色彩空间，UI 颜色将不准确，因为应用不支持 UI 中的颜色管理。