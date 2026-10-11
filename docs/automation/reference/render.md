# Render commands

## `render.preview`

<!-- BEGIN GENERATED: render.preview -->
Changes state: no.

Renders the open image with the current edits and writes <file_stem>_current.png into out_dir. With compare 'root' (the default), it also renders the image as imported (the root) and writes <file_stem>_root.png; with compare 'none', it renders and writes only the current file. file_stem defaults to the photo file name without its extension. The long edge of each file is long_edge pixels at most. 'region' (x, y, w, h in 0..1 of the cropped image) renders only that part. Answers -32005 while the editor renders another image job, for example an open comparison; that job continues.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `compare` | `none` \| `root` | no | `"root"` | 'root' also writes the imported image. |
| `file_stem` | string | no |  | First part of the file names. Default: the photo file name without its extension. |
| `long_edge` | integer | no | `2048` | 16 to 4096. |
| `out_dir` | string | yes |  | Folder of the PNG files. |
| `region` | object | no |  |  |
| `region.h` | number | yes |  | 0 to 1. |
| `region.w` | number | yes |  | 0 to 1. |
| `region.x` | number | yes |  | 0 to 1. |
| `region.y` | number | yes |  | 0 to 1. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `current` | object | yes |  |
| `current.height` | integer | yes |  |
| `current.path` | string | yes |  |
| `current.width` | integer | yes |  |
| `root` | object | no |  |
| `root.height` | integer | yes |  |
| `root.path` | string | yes |  |
| `root.width` | integer | yes |  |
<!-- END GENERATED: render.preview -->

The PNG files are display-referred 8-bit sRGB images of the current edits, the same look as the
editor viewport. Use them to check an edit, or give them to an image analysis tool. Use
`export.run` for the final files.

**Errors**

- `-32602` `/region/<name>`: `region` takes `x`, `y`, `w`, and `h`, each 0 to 1.
- `-32602` `/region`: the region is empty or goes past the edge of the image.
- `-32602` `/file_stem`: `the file stem must not contain '/', '\', or ':'`.
- `-32001`: `No editor image is open.`
- `-32002`: the editor is not interactive (`Preview images require an interactive session`).
- `-32005`: the editor renders another image job.
- `-32007`: the render or the file write failed.

**Example**

```bash
alcedo-cli --json call render.preview '{"out_dir":"D:/Photos/previews","long_edge":1024}'
```

```json
{"current":{"height":1024,"path":"D:/Photos/previews/DSC0027_current.png","width":682},"root":{"height":1024,"path":"D:/Photos/previews/DSC0027_root.png","width":682}}
```

A detail of the center, without the root image:

```bash
alcedo-cli --json call render.preview '{"out_dir":"D:/Photos/previews","long_edge":512,"compare":"none","file_stem":"detail","region":{"x":0.25,"y":0.25,"w":0.5,"h":0.5}}'
```

```json
{"current":{"height":512,"path":"D:/Photos/previews/detail_current.png","width":341}}
```
