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

2. Make sure the Windows GLib2 dependency is available.

When you configure Pu-erh Lab with the repository's vcpkg toolchain, the bundled Lensfun build will first try to use:

- `vcpkg/installed/<triplet>`

If `glib` is not installed there yet, install it first:

```powershell
.\vcpkg\vcpkg.exe install glib:x64-windows
```

If you want to use a non-vcpkg GLib2 package, pass `PUERHLAB_LENSFUN_GLIB2_BASE_DIR` explicitly when configuring Pu-erh Lab.

## Example Top-Level Configure

```powershell
cmd /c scripts\msvc_env.cmd --preset win_debug `
  -DCMAKE_PREFIX_PATH="D:/Qt/6.9.3/msvc2022_64/lib/cmake" `
  -Deasy_profiler_DIR="$PWD/pu-erh_lab/third_party/easy_profiler-v2.1.0-msvc15-win64/lib/cmake/easy_profiler"
```

Then build Pu-erh Lab as usual:

```powershell
cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 4
```

If the bundled Lensfun configure step still fails on Windows, verify the Lensfun submodule exists and that either the vcpkg GLib2 headers or your override path exists:

```powershell
Test-Path .\pu-erh_lab\src\third_party\lensfun\CMakeLists.txt
Test-Path .\vcpkg\installed\x64-windows\include\glib-2.0\glib.h
# or, if you passed an override:
Test-Path <your-glib2-path>
```
