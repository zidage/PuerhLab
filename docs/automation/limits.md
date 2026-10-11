# Known limits

Behavior to know about in Alcedo Studio 0.3.2 (protocol version 1), and the open work.

## Installation

- **The packages do not contain `alcedo-cli` yet.** The Windows installer and the macOS app have
  only `alcedo_main`. Build `alcedo-cli` from source (target `alcedo-cli`) until the packages have
  it ([#329](https://github.com/zidage/AlcedoStudio/issues/329)).
- **macOS app:** the host program in the app is `AlcedoStudio.app/Contents/MacOS/AlcedoStudio`,
  and `session start` does not look for it there. Give `--host-binary` (see
  [Installation](installation.md#where-the-programs-are)).
- **No switch in the settings page** for the GUI session. Change `automation/serverEnabled` in the
  settings store (see [Installation](installation.md#the-gui-session-setting)).

## Library

- **Import takes RAW files only.** JPEG, PNG, TIFF, and other files count as `unsupported`.
- **`library.thumbnail` shows committed edits, through a cache.** A thumbnail made just after an
  edit can show the earlier edits. Use `render.preview` for the open photo.
- **No command copies or pastes adjustments** between photos. Set the same fields on each photo
  with `editor.batch_set`.

## Editor

- **One photo at a time.** `editor.open` saves and replaces the open photo. `project.save` and
  `editor.close` also close the editor image.
- **A failed editor save blocks the editor.** After a failed save the editor state is
  `RetainedImageFailure`: edits fail with `-32002`, and `editor.close` and `project.close` fail
  with `-32007` ("Resolve the failed editor save before closing the editor"). No command recovers
  from it; in a headless session, stop the session process (`session stop`, or end the process)
  and start again. The project keeps the edits of the last successful save.
- **`editor.batch_set` is not atomic.** When a change fails, the changes before it stay (see
  [editor.batch_set](reference/editor.md#editorbatch_set)).
- **No local adjustments.** Masks and brushes have no command.
- **Rotation shrinks the crop.** The editor keeps the rotated crop inside the image, so after a
  rotation the crop rectangle is smaller than you sent. Read the returned value.

## Render and export

- **One image job at a time.** `render.preview` answers `-32005` while the editor renders another
  image job, for example an open comparison in the GUI. Try again after it ends.
- **One export at a time.** A second `export.run` answers `-32005` until the first export ends.

## GUI session

- **The user can take control at any time.** Every command that changes state can fail with
  `-32004`. An agent must stop and wait for **Return control**; it must not retry in a loop.
- **`session stop` on the `gui` session closes the application.**
- **No test runs random command sequences** against a session yet
  ([#330](https://github.com/zidage/AlcedoStudio/issues/330)).

## Protocol

- **No MCP server yet.** Agents call `alcedo-cli` from a shell
  ([#329](https://github.com/zidage/AlcedoStudio/issues/329)).
- **Local only.** The socket accepts connections from the same OS user on the same computer.
