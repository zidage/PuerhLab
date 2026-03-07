---
name: puerhlab-msvc-cmake
description: Use when working on pu-erh_lab with CMake on Windows/MSVC, especially when the user mentions MSVC, Windows, presets, Ninja, CUDA, or scripts/msvc_env.cmd, or when Codex would otherwise run bare cmake commands in this repository.
---

# Pu-erh Lab MSVC CMake

Use this skill for configure, build, install, and test workflows in this repository on Windows.

## Workflow

- Run commands from the repository root.
- Prefer the presets in `CMakePresets.json`: `win_debug` and `win_release`.
- Do not invoke bare `cmake` directly for configure/build/install when `scripts/msvc_env.cmd` is available.
- Use `cmd /c scripts\msvc_env.cmd ...` so Visual Studio, MSVC, and CUDA environment variables are initialized first.

## Command Templates

- Configure debug: `cmd /c scripts\msvc_env.cmd --preset win_debug`
- Configure release: `cmd /c scripts\msvc_env.cmd --preset win_release`
- Build debug: `cmd /c scripts\msvc_env.cmd --build --preset win_debug --parallel 4`
- Build release: `cmd /c scripts\msvc_env.cmd --build --preset win_release --parallel 4`
- Install release: `cmd /c scripts\msvc_env.cmd --install build/release --prefix build/install`
- Run tests after a debug build: `ctest --test-dir build/debug --output-on-failure`

## Rules

- Append user-provided `-D...` cache entries to the configure command after the preset.
- If the user asks for a `cmake --build build/...` style command, translate it to the wrapper form instead of changing the build intent.
- If the wrapper fails before reaching CMake, inspect `scripts/msvc_env.cmd` before changing presets or toolchain arguments.
