# Lensfun 内置构建说明（Windows）

Pu-erh Lab 现在会直接从上游子模块构建 Lensfun，源码位置在：

- `pu-erh_lab/src/third_party/lensfun`

现在不需要再单独配置或安装 Lensfun。顶层构建会在当前 CMake 构建目录下自动驱动一个独立的 Lensfun out-of-source 构建：

- `build/<preset>/third_party/lensfun`

## 必要准备

1. 初始化子模块：

```powershell
git submodule update --init --recursive pu-erh_lab/src/third_party/lensfun
```

2. 准备上游 Lensfun 在 Windows 下需要的 GLib2 包。

Pu-erh Lab 默认会把下面这个路径转发给 Lensfun：

- `pu-erh_lab/third_party/glib-2.28.1`

如果你的 GLib2 不在这个位置，请在配置 Pu-erh Lab 时传入 `PUERHLAB_LENSFUN_GLIB2_BASE_DIR`。

## 顶层配置示例

```powershell
cmd /c scripts\msvc_env.cmd --preset win_debug `
  -DCMAKE_PREFIX_PATH="D:/Qt/6.9.3/msvc2022_64/lib/cmake" `
  -Deasy_profiler_DIR="$PWD/pu-erh_lab/third_party/easy_profiler-v2.1.0-msvc15-win64/lib/cmake/easy_profiler" `
  -DPUERHLAB_LENSFUN_GLIB2_BASE_DIR="$PWD/pu-erh_lab/third_party/glib-2.28.1"
```

然后照常构建 Pu-erh Lab：

```powershell
cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 4
```

如果 Windows 下 bundled Lensfun 的配置阶段失败，请先确认这两个路径存在：

```powershell
Test-Path .\pu-erh_lab\src\third_party\lensfun\CMakeLists.txt
Test-Path .\pu-erh_lab\third_party\glib-2.28.1
```
