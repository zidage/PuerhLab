@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ---- Locate VS installation via vswhere
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo [cmake_msvc] vswhere not found: "%VSWHERE%"
  exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
  set "VSINSTALL=%%i"
)

if not defined VSINSTALL (
  echo [cmake_msvc] Failed to locate Visual Studio installation.
  exit /b 1
)

rem ---- Setup MSVC environment (x64)
set "VSDEVCMD=%VSINSTALL%\Common7\Tools\VsDevCmd.bat"
if exist "%VSDEVCMD%" (
  call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul
) else (
  set "VCVARSALL=%VSINSTALL%\VC\Auxiliary\Build\vcvarsall.bat"
  if not exist "%VCVARSALL%" (
    echo [cmake_msvc] Neither VsDevCmd.bat nor vcvarsall.bat found under: %VSINSTALL%
    exit /b 1
  )
  call "%VCVARSALL%" x64 >nul
)

rem ---- Find the real cmake.exe
set "REAL_CMAKE="
for /f "usebackq tokens=*" %%p in (`where cmake 2^>nul`) do (
  set "REAL_CMAKE=%%p"
  goto :found
)
:found

if not defined REAL_CMAKE (
  echo [cmake_msvc] cmake.exe not found in PATH after environment setup.
  exit /b 1
)

rem ---- Forward all arguments to real cmake
"%REAL_CMAKE%" %*
exit /b %ERRORLEVEL%
