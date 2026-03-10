# Bundled Lensfun Build Notes (Windows)

Pu-erh Lab now builds Lensfun directly from the upstream submodule at:

- `pu-erh_lab/src/third_party/lensfun`

You do not need to configure or install Lensfun separately anymore. The top-level build drives a dedicated out-of-source Lensfun build under your active CMake build tree:

- `build/<preset>/third_party/lensfun`

## Required Setup

1. Initialize the submodule:

```powershell
git submodule update --init --recursive pu-erh_lab/src/third_party/lensfun
```

2. Provide the Windows GLib2 package expected by the upstream Lensfun build.

By default, Pu-erh Lab forwards this path to Lensfun:

- `pu-erh_lab/third_party/glib-2.28.1`

If your GLib2 package lives elsewhere, pass `PUERHLAB_LENSFUN_GLIB2_BASE_DIR` when configuring Pu-erh Lab.

## Example Top-Level Configure

```powershell
cmd /c scripts\msvc_env.cmd --preset win_debug `
  -DCMAKE_PREFIX_PATH="D:/Qt/6.9.3/msvc2022_64/lib/cmake" `
  -Deasy_profiler_DIR="$PWD/pu-erh_lab/third_party/easy_profiler-v2.1.0-msvc15-win64/lib/cmake/easy_profiler" `
  -DPUERHLAB_LENSFUN_GLIB2_BASE_DIR="$PWD/pu-erh_lab/third_party/glib-2.28.1"
```

Then build Pu-erh Lab as usual:

```powershell
cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 4
```

If the bundled Lensfun configure step fails on Windows, verify both of these paths exist:

```powershell
Test-Path .\pu-erh_lab\src\third_party\lensfun\CMakeLists.txt
Test-Path .\pu-erh_lab\third_party\glib-2.28.1
```
