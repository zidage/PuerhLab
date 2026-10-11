# Export commands

## `export.options`

<!-- BEGIN GENERATED: export.options -->
Changes state: no.

Returns the export defaults, the formats, the bit depths of each format, the TIFF compressions, the resize modes, the print units, and the limits of the size and quality values. export.run uses the same rules as the export panel.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `bit_depths` | object | yes |  |
| `defaults` | object | yes |  |
| `formats` | array | yes |  |
| `limits` | object | yes |  |
| `print_units` | array | yes |  |
| `resize_modes` | array | yes |  |
| `tiff_compressions` | array | yes |  |
<!-- END GENERATED: export.options -->

The defaults are the values of the export dialog of this user.

**Errors:** none.

**Example**

```bash
alcedo-cli --json call export.options
```

```json
{"bit_depths":{"EXR":[16,32],"JPEG":[8],"PNG":[8,16],"TIFF":[8,16,32]},"defaults":{"bit_depth":8,"dpi":300,"embed_icc_profile":true,"file_name_pattern":"{source}","format":"JPEG","height":2048,"include_metadata":true,"long_edge":2048,"out_dir":"C:/Users/me/Pictures","png_level":5,"print_height":297,"print_unit":"mm","print_width":210,"quality":95,"resize_mode":"original","subfolder":"","tiff_compression":"NONE","ultra_hdr_dither":true,"ultra_hdr_quality":95,"width":2048},"formats":["JPEG","PNG","TIFF","EXR"],"limits":{"dpi":{"max":10000,"min":1},"height":{"max":100000,"min":1},"long_edge":{"max":16384,"min":256}, ...},"print_units":["mm","cm","in"],"resize_modes":["original","long_edge","bounds","print"], ...}
```

## `export.run`

<!-- BEGIN GENERATED: export.run -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Exports the photos 'element_ids' into 'out_dir' with the given options (see export.options; an option that is not given has its default). An option that breaks a rule is rejected with the allowed values and no export starts. Returns the task id for tasks.wait as soon as the export runs; the finished task has the counts, and export.status lists the state of each photo.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `bit_depth` | integer | no |  | JPEG 8; PNG 8, 16; TIFF 8, 16, 32; EXR 16, 32. |
| `dpi` | integer | no |  | Print resolution, 1 to 10000. |
| `element_ids` | array of integer | yes |  | at least 1 item(s); each item at least 1. The photos to export. |
| `embed_icc_profile` | boolean | no |  | Embed the output ICC profile. |
| `file_name_pattern` | string | no |  | File name pattern, for example {source} or {date:yyyy MM dd}-{sequence:0000}. Fields: source, date, sequence, cameraMake, cameraModel, lens, iso, aperture, shutter, focal, rating. |
| `format` | `JPEG` \| `PNG` \| `TIFF` \| `EXR` | no |  | Output format of SDR photos. HDR photos export as UltraHDR JPEG. |
| `height` | integer | no |  | Bounds height in pixels, 1 to 100000. |
| `include_metadata` | boolean | no |  | Write the photo metadata. |
| `long_edge` | integer | no |  | Long edge in pixels, 256 to 16384. |
| `out_dir` | string | yes |  | The output folder. It is created when it does not exist. |
| `png_level` | integer | no |  | PNG compression level, 0 to 9. |
| `print_height` | number | no |  | Print height, more than 0 and at most 100000. |
| `print_unit` | `mm` \| `cm` \| `in` | no |  | Unit of the print size. |
| `print_width` | number | no |  | Print width, more than 0 and at most 100000. |
| `quality` | integer | no |  | JPEG quality, 1 to 100. |
| `resize_mode` | `original` \| `long_edge` \| `bounds` \| `print` | no |  | original, long_edge (long_edge), bounds (width, height), or print (print_width, print_height, print_unit, dpi). Without it, the first size value given selects the mode. |
| `subfolder` | string | no |  | Write into this subfolder of out_dir. |
| `tiff_compression` | `NONE` \| `LZW` \| `ZIP` | no |  | TIFF compression. |
| `ultra_hdr_dither` | boolean | no |  | Dither the UltraHDR gain map. |
| `ultra_hdr_quality` | integer | no |  | UltraHDR JPEG quality of HDR photos, 1 to 100. |
| `width` | integer | no |  | Bounds width in pixels, 1 to 100000. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `queued` | integer | yes |  |
| `skipped` | integer | yes |  |
| `task_id` | string | yes |  |
<!-- END GENERATED: export.run -->

The export uses the saved edits of each photo. Close the editor (`editor.close`) or save
(`project.save`) first, so that the last edits are saved.

**Errors**

- `-32602` `/element_ids`: unknown ids; `data.unknown_ids` lists them. Nothing is exported.
- `-32602` `/<option>`: an option breaks a rule, for example
  `bit_depth for JPEG must be one of: 8` or `long_edge must be in [256, 16384]`.
- `-32005`: `An export is already running.` Wait with `tasks.wait`, then try again.
- `-32002`: the export cannot start; the message tells why, for example when no photo can be
  queued.

**Example**

```bash
alcedo-cli --json call export.run '{"element_ids":[1,2],"out_dir":"D:/Photos/export","format":"JPEG","quality":90,"long_edge":2048}'
alcedo-cli --json call tasks.wait '{"task_id":"export-1"}'
```

```json
{"queued":2,"skipped":0,"task_id":"export-1"}
{"counts":{"completed":2,"failed":0,"skipped":0,"succeeded":2,"total":2},"kind":"export","message":"Export complete. Written 2/2 image(s), failed 0.","progress":100,"state":"succeeded","task_id":"export-1"}
```

A 16-bit TIFF with ZIP compression in a subfolder:

```bash
alcedo-cli --json call export.run '{"element_ids":[1],"out_dir":"D:/Photos/export","format":"TIFF","bit_depth":16,"tiff_compression":"ZIP","subfolder":"tiff"}'
```

```json
{"queued":1,"skipped":0,"task_id":"export-2"}
```

This writes `D:/Photos/export/tiff/DSC0027.tif`.

## `export.status`

<!-- BEGIN GENERATED: export.status -->
Changes state: no.

Returns the state of the running or last export: the task id, the status message, the counts, the error summary, and the state of each photo. 'queue' lists the export queue of the export panel (empty in the headless host).

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `counts` | object | yes |  |
| `error_summary` | string | yes |  |
| `in_flight` | boolean | yes |  |
| `items` | array | yes |  |
| `message` | string | yes |  |
| `queue` | array | yes |  |
| `task_id` | string | yes |  |
<!-- END GENERATED: export.status -->

**Errors:** none.

**Example**

```bash
alcedo-cli --json call export.status
```

```json
{"counts":{"completed":2,"failed":0,"skipped":0,"succeeded":2,"total":2},"error_summary":"","in_flight":false,"items":[{"element_id":1,"image_id":1,"state":"succeeded"},{"element_id":2,"image_id":2,"state":"succeeded"}],"message":"Export complete. Written 2/2 image(s), failed 0.","queue":[],"task_id":"export-1"}
```
