# Lensfun 本地构建指南（Windows）

Pu-erh Lab 要求从 `pu-erh_lab/third_party/lensfun` 本地构建 Lensfun，并安装到：

- `pu-erh_lab/third_party/lensfun/install`

顶层构建通过 `PUERHLAB_LENSFUN_ROOT` 读取该目录，并期望存在：

- `include/lensfun/lensfun.h`
- `lib/lensfun.lib`
- `bin/lensfun.dll`（配置阶段可选，运行时必需）
- `share/lensfun/version_2/*.xml`（Lensfun 数据库文件）

## 使用自定义 Lensfun CMakeLists

请使用仓库内的：

- `pu-erh_lab/third_party/lensfun/CMakeLists.txt`

针对 Pu-erh Lab，建议在该 CMakeLists 下使用以下参数：

- `BUILD_STATIC=OFF`
- `BUILD_TESTS=OFF`
- `BUILD_LENSTOOL=OFF`
- `BUILD_DOC=OFF`
- `CMAKE_INSTALL_PREFIX=<repo>/pu-erh_lab/third_party/lensfun/install`

## 需要手动准备的 third_party 目录

Lensfun 自定义 CMake 会将 `GLIB2_BASE_DIR` 默认指向：

- `pu-erh_lab/third_party/glib-2.28.1`

这个 `glib-2.28.1` 目录是 Windows 下构建 Lensfun 的必需依赖，但仓库未包含该包，需要你手动放置到 `third_party`（而不是依赖 Pu-erh Lab 的 vcpkg 流程自动提供）。

## 构建步骤

在仓库根目录（`D:\Projects\pu-erh_lab`）执行：

```powershell
# 使用自定义 CMakeLists.txt 配置 Lensfun
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

# 编译并安装 Lensfun
.\scripts\msvc_env.cmd --build pu-erh_lab\third_party\lensfun\build-puerhlab --parallel
.\scripts\msvc_env.cmd --install pu-erh_lab\third_party\lensfun\build-puerhlab --config Release
```

如果你的 GLib2 放在其他位置，请调整 `-DGLIB2_BASE_DIR=<path>`。

## 在构建 Pu-erh Lab 前检查

```powershell
Test-Path .\pu-erh_lab\third_party\lensfun\install\include\lensfun\lensfun.h
Test-Path .\pu-erh_lab\third_party\lensfun\install\lib\lensfun.lib
Test-Path .\pu-erh_lab\third_party\lensfun\install\bin\lensfun.dll
```

然后在 Pu-erh Lab 配置时传入：

```powershell
-DPUERHLAB_LENSFUN_ROOT="$PWD/pu-erh_lab/third_party/lensfun/install"
```
