# Installation

`alcedo-cli` is a small program beside `alcedo_main`. It needs no installation of its own. It
finds `alcedo_main` in its own folder when it starts a headless session.

## Where the programs are

| Build | `alcedo-cli` | `alcedo_main` |
| --- | --- | --- |
| Windows build from source | `build/<preset>/alcedo_studio/src/alcedo-cli.exe` | `build/<preset>/alcedo_studio/src/alcedo_main.exe` |
| macOS build from source | `build/<preset>/alcedo_studio/src/alcedo-cli` | `build/<preset>/alcedo_studio/src/alcedo_main.app/Contents/MacOS/alcedo_main` |
| Windows installer | Not in the package yet ([#329](https://github.com/zidage/AlcedoStudio/issues/329)). | `<install folder>\bin\alcedo_main.exe`; the default install folder is `C:\Program Files\Alcedo Studio` |
| macOS disk image | Not in the package yet ([#329](https://github.com/zidage/AlcedoStudio/issues/329)). | `AlcedoStudio.app/Contents/MacOS/AlcedoStudio` |

`<preset>` is the CMake preset, for example `release` or `debug`. See
[Build from source](../build_from_source.md).

`session start` looks for `alcedo_main` in this order:

1. `--host-binary <path>`.
2. The folder of `alcedo-cli`.
3. `../MacOS/alcedo_main` (for `alcedo-cli` in `Contents/Helpers` of the app bundle).
4. `alcedo_main.app/Contents/MacOS/alcedo_main` beside `alcedo-cli` (a macOS build tree).

The packaged macOS program is named `AlcedoStudio`, so with a packaged app give
`--host-binary AlcedoStudio.app/Contents/MacOS/AlcedoStudio`.

A GUI session needs no `alcedo-cli` path: the Alcedo Studio window starts its own session.

## Put alcedo-cli on the PATH

The examples in these pages call `alcedo-cli` by name. Add its folder to `PATH`, or call it
with its full path.

```bash
export PATH="/d/Projects/pu-erh_lab/build/release/alcedo_studio/src:$PATH"
```

## Check the setup

```bash
alcedo-cli help
alcedo-cli session list
```

`session list` prints `no session files in '<dir>'` when no session runs. When Alcedo Studio is
open, it prints the `gui` session:

```text
gui	live	gui	pid 36984	D:/Photos/trip.alcd
```

## The GUI session setting

The Alcedo Studio window starts the session `gui` when the setting `automation/serverEnabled` is
true. This is the default. The socket accepts connections only from the same OS user. To turn
the server off, set `automation/serverEnabled` to `false` in the Alcedo Studio settings and
restart the application. The settings page has no switch for it yet. The settings are in the
registry key `HKEY_CURRENT_USER\Software\Alcedo\Alcedo` on Windows and in
`~/Library/Preferences/app.alcedo.Alcedo.plist` on macOS.

```bash
reg add "HKCU\Software\Alcedo\Alcedo\automation" /v serverEnabled /t REG_SZ /d false /f
defaults write app.alcedo.Alcedo automation.serverEnabled -bool false
```

## Shell quoting

The parameters of `call` are one JSON object in one argument.

| Shell | Example |
| --- | --- |
| bash, zsh, Git Bash | `alcedo-cli call editor.get '{"fields":["exposure"]}'` |
| PowerShell 7.3 or later | `alcedo-cli call editor.get '{"fields":["exposure"]}'` |
| Windows PowerShell 5.1 | `alcedo-cli call editor.get '{\"fields\":[\"exposure\"]}'` |
| cmd.exe | `alcedo-cli call editor.get "{\"fields\":[\"exposure\"]}"` |
