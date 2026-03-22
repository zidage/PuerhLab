@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ---- Locate VS installation via vswhere
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo [cargo_msvc] vswhere not found: "%VSWHERE%"
  exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
  set "VSINSTALL=%%i"
)

if not defined VSINSTALL (
  echo [cargo_msvc] Failed to locate Visual Studio installation.
  exit /b 1
)

rem ---- Setup MSVC environment (x64)
set "VSDEVCMD=%VSINSTALL%\Common7\Tools\VsDevCmd.bat"
if exist "%VSDEVCMD%" (
  call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul
) else (
  set "VCVARSALL=%VSINSTALL%\VC\Auxiliary\Build\vcvarsall.bat"
  if not exist "%VCVARSALL%" (
    echo [cargo_msvc] Neither VsDevCmd.bat nor vcvarsall.bat found under: %VSINSTALL%
    exit /b 1
  )
  call "%VCVARSALL%" x64 >nul
)

rem ---- Find the real cargo.exe
set "THIS_SCRIPT=%~f0"
set "REAL_CARGO="
for /f "usebackq tokens=*" %%p in (`where cargo 2^>nul`) do (
  if /I not "%%~fp"=="%THIS_SCRIPT%" (
    set "REAL_CARGO=%%~fp"
    goto :found
  )
)
:found

if not defined REAL_CARGO (
  if exist "%USERPROFILE%\.cargo\bin\cargo.exe" (
    set "REAL_CARGO=%USERPROFILE%\.cargo\bin\cargo.exe"
  )
)

if not defined REAL_CARGO (
  echo [cargo_msvc] cargo.exe not found in PATH after environment setup.
  exit /b 1
)

rem ---- Forward all arguments to real cargo
"%REAL_CARGO%" %*
exit /b %ERRORLEVEL%
