# Local Lensfun Build Guide (Windows)

Pu-erh Lab expects Lensfun to be built locally from `pu-erh_lab/third_party/lensfun` and installed into:

- `pu-erh_lab/third_party/lensfun/install`

The top-level build reads this location through `PUERHLAB_LENSFUN_ROOT` and expects:

- `include/lensfun/lensfun.h`
- `lib/lensfun.lib`
- `bin/lensfun.dll` (optional at configure time, required at runtime)
- `share/lensfun/version_2/*.xml` (Lensfun database files)

## Use the Custom Lensfun CMakeLists

Use the vendored file:

- `pu-erh_lab/third_party/lensfun/CMakeLists.txt`

For Pu-erh Lab, configure Lensfun with these options from that CMake file:

- `BUILD_STATIC=OFF`
- `BUILD_TESTS=OFF`
- `BUILD_LENSTOOL=OFF`
- `BUILD_DOC=OFF`
- `CMAKE_INSTALL_PREFIX=<repo>/pu-erh_lab/third_party/lensfun/install`

## Missing Third-Party Folder You Must Provide

The Lensfun custom CMake defaults `GLIB2_BASE_DIR` to:

- `pu-erh_lab/third_party/glib-2.28.1`

This `glib-2.28.1` package is required for building Lensfun on Windows, is not committed in this repository, and should be provided as a local third-party folder (instead of relying on Pu-erh Lab's vcpkg dependency flow).

## Build Steps

From repository root (`D:\Projects\pu-erh_lab`):

```powershell
# Configure Lensfun with the custom CMakeLists.txt
.\scripts\msvc_env.cmd -S pu-erh_lab\third_party\lensfun -B pu-erh_lab\third_party\lensfun\build-puerhlab -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DBUILD_STATIC=OFF `
  -DBUILD_TESTS=OFF `
  -DBUILD_LENSTOOL=OFF `
  -DBUILD_DOC=OFF `
  -DGLIB2_BASE_DIR="$PWD/pu-erh_lab/third_party/glib-2.28.1" `
  -DCMAKE_INSTALL_PREFIX="$PWD/pu-erh_lab/third_party/lensfun/install"

# Build + install Lensfun
.\scripts\msvc_env.cmd --build pu-erh_lab\third_party\lensfun\build-puerhlab --parallel
.\scripts\msvc_env.cmd --install pu-erh_lab\third_party\lensfun\build-puerhlab --config Release
```

If your GLib2 package is stored elsewhere, adjust `-DGLIB2_BASE_DIR=<path>`.

## Verify Before Building Pu-erh Lab

```powershell
Test-Path .\pu-erh_lab\third_party\lensfun\install\include\lensfun\lensfun.h
Test-Path .\pu-erh_lab\third_party\lensfun\install\lib\lensfun.lib
Test-Path .\pu-erh_lab\third_party\lensfun\install\bin\lensfun.dll
```

Then configure Pu-erh Lab with:

```powershell
-DPUERHLAB_LENSFUN_ROOT="$PWD/pu-erh_lab/third_party/lensfun/install"
```

