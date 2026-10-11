# Workflows

These examples were run on the release build with three sample RAW files. The outputs are real;
long outputs are shortened with `...`. The paths are examples: use your own folders. In
PowerShell and cmd, quote the JSON as [Installation](installation.md#shell-quoting) shows.

## Start a headless session

```bash
alcedo-cli --session trip session start --headless --create D:/Photos,trip
```

```text
session 'trip' started (process 42892, project D:/Photos/trip.alcd)
```

`--session trip` names the session; without it the session is named `default`.
`--create D:/Photos,trip` creates the project `D:/Photos/trip.alcd`; `--project <path>` opens an
existing one. With one live session, every command goes to it. With more, give `--session` or
set `ALCEDO_SESSION`.

## Import a folder and rate photos

Import runs as a task. Wait for it before you list the photos.

```bash
alcedo-cli call library.import '{"folder":"D:/Photos/2026-trip"}'
alcedo-cli call tasks.wait '{"task_id":"import-1"}'
```

```json
{"counts":{"completed":3,"excluded":0,"failed":0,"total":3,"unsupported":0},"kind":"import","message":"Import complete: 3 imported, 0 not selected, 0 unsupported, 0 failed","progress":100,"state":"succeeded","task_id":"import-1"}
```

Import takes RAW files only; other files count as `unsupported`. Add `"recursive":true` for the
subfolders.

```bash
alcedo-cli call library.list '{"sort":"capture_time"}'
```

```json
{"items":[{"element_id":1,"file_name":"DSC0027.ARW","image_id":1,"rating":0},{"element_id":2,"file_name":"DSC00800.ARW","image_id":2,"rating":0},{"element_id":3,"file_name":"DSC06683.dng","image_id":3,"rating":0}],"offset":0,"total":3}
```

Use the `element_id` in the other commands. To rate photos:

```bash
alcedo-cli call library.rate '{"element_ids":[1,3],"rating":4}'
alcedo-cli call tasks.wait '{"idle":true}'
alcedo-cli call library.list '{"sort":"rating","descending":true,"limit":2}'
```

```json
{"rated_count":2,"rating":4,"task_id":"bgtask-1"}
{"items":[{"element_id":1,"file_name":"DSC0027.ARW","image_id":1,"rating":4},{"element_id":3,"file_name":"DSC06683.dng","image_id":3,"rating":4}],"offset":0,"total":3}
```

## Edit a photo: preview, look, adjust

The editor edits one photo at a time. Each change is a commit in the history of the active
Version, and the editor saves it.

1. Open the photo and read the values that you want to change:

   ```bash
   alcedo-cli call editor.open '{"element_id":1}'
   alcedo-cli call editor.get '{"fields":["exposure","color_temp","crop_rotate"]}'
   ```

   ```json
   {"element_id":1,"head_commit":null,"image_id":1,"state":"Interactive","values":{"color_temp":{"kelvin":5121.59033203125,"mode":"as_shot","tint":14.783473014831543},"crop_rotate":{"angle_degrees":0,"aspect_height":1,"aspect_preset":"free","aspect_width":1,"height":1,"width":1,"x":0,"y":0},"exposure":1.5}}
   ```

   A photo without edits has the default look: exposure +1.5, contrast 15, saturation 30.

2. Render a preview. It is the display-referred image that the user sees.

   ```bash
   alcedo-cli call render.preview '{"out_dir":"D:/Photos/previews","compare":"none","long_edge":1024,"file_stem":"step0"}'
   ```

   ```json
   {"current":{"height":1024,"path":"D:/Photos/previews/step0_current.png","width":682}}
   ```

3. Look at the preview and decide what to change: for example the brightness, the white
   balance, or a tilted horizon. An agent with image input opens the PNG file and looks at it.
   [Editor fields](reference/editor_fields.md) tells what each field does, for example which
   direction of `kelvin` makes the image warmer and which sign of `angle_degrees` turns it
   clockwise.

4. Change the values. `editor.batch_set` makes one commit per change; `note` marks them in the
   history.

   ```bash
   alcedo-cli call editor.batch_set '{"changes":[{"field":"exposure","value":1.7},{"field":"color_temp","value":{"kelvin":5400}}],"note":"agent: exposure and white balance"}'
   ```

   ```json
   {"applied":2,"head_commit":"7060ad9cfd466e5caa51a66f8c0750af","values":{"color_temp":{"kelvin":5400,"mode":"custom","tint":14.783473014831543},"exposure":1.7000000476837158}}
   ```

   A write with `kelvin` or `tint` sets the `color_temp` mode to `custom`.

5. Render a new preview, look again, and repeat until the photo looks right. Small steps are
   easier to judge than one large jump.

To compare with the photo as imported, render with `"compare":"root"`: it also writes
`<file_stem>_root.png`. To look at a detail, give `region` (`x`, `y`, `w`, `h` in 0 to 1 of the
cropped image).

### Measure a preview (optional)

When you want numbers, for example to compare the brightness of a series, the script
[`analyze_preview.py`](https://github.com/zidage/alcedo-skills/blob/main/skills/alcedo/scripts/analyze_preview.py)
of the alcedo skill measures a preview. It needs Python 3.9 with numpy and Pillow. It reports the
L* percentiles and clipping, the color of the most neutral pixels, and the tilt of the strongest
straight lines (counterclockwise positive: add it to `angle_degrees` to level the photo):

```bash
python analyze_preview.py D:/Photos/previews/step0_current.png
```

```text
Tone:
  L* p1/p5/p50/p95/p99: 12.7 / 20.0 / 47.3 / 94.9 / 95.6  (key: normal)
  clipped highlights 0.00%, clipped shadows 0.00%
  suggested exposure change +0.18 EV
Color:
  nearly grey pixels: a* -0.72, b* -2.44 -> cast: neutral (reliability high)
  mean chroma 24.6, very saturated 1.16%
Tilt:
  -4.0 degrees from vertical lines (counterclockwise positive), confidence 0.4 (55870 edge pixels)
  horizontal lines +0.4 (prominence 1.39), vertical lines -4.0 (prominence 2.09)
  To level the photo, add the tilt to crop_rotate.angle_degrees.
```

The numbers describe the image; they do not say whether it looks good. A dark night photo or a
warm sunset can be right as it is.


## History, undo, and redo

```bash
alcedo-cli call editor.history
```

```json
{"can_redo":false,"can_undo":true,"commits":[{"author":"agent","commit_hash":"7060ad9cfd466e5caa51a66f8c0750af","field":"color_temp","note":"agent: exposure and white balance","operation_kind":"set_parameter","parent_hash":"44d3248d83b60e187fe9373d59f9e186","position":"current"},{"author":"agent","commit_hash":"44d3248d83b60e187fe9373d59f9e186","field":"exposure","note":"agent: exposure and white balance","operation_kind":"set_parameter","parent_hash":null,"position":"applied"}],"head_commit":"7060ad9cfd466e5caa51a66f8c0750af","version_id":"3fdfa0facee012d4f8f233b6a9bea1f0"}
```

The newest commit is first. `editor.undo` moves the head back one commit; `editor.redo` moves it
forward:

```bash
alcedo-cli call editor.undo
alcedo-cli call editor.redo
```

```json
{"can_redo":true,"can_undo":true,"head_commit":"44d3248d83b60e187fe9373d59f9e186"}
{"can_redo":false,"can_undo":true,"head_commit":"7060ad9cfd466e5caa51a66f8c0750af"}
```

A new edit after an undo removes the commits after the head.

## Try alternatives with Versions

A photo has one or more Versions; each has its own history. Edit the active Version for normal
work. Use Versions to keep alternatives.

- `versions.branch` copies the edits up to a commit (default: the head) into a new Version:

  ```bash
  alcedo-cli call versions.branch '{"name":"Exposure only","from_commit":"44d3248d83b60e187fe9373d59f9e186"}'
  ```

  ```json
  {"head_commit":"44d3248d83b60e187fe9373d59f9e186","name":"Exposure only","version_id":"370bfb1620bfbcc0b57417f0cdaee650"}
  ```

- `versions.create` starts a new Version at the photo as imported, with no edits:

  ```bash
  alcedo-cli call versions.create '{"name":"Warm"}'
  ```

  ```json
  {"head_commit":null,"name":"Warm","version_id":"2384b25e60ee5642099b6773b67c256f"}
  ```

- `versions.list` lists them; `versions.checkout` makes one active (the id is a string):

  ```bash
  alcedo-cli call versions.checkout '{"version_id":"3fdfa0facee012d4f8f233b6a9bea1f0"}'
  ```

  ```json
  {"head_commit":"7060ad9cfd466e5caa51a66f8c0750af","name":"Default","version_id":"3fdfa0facee012d4f8f233b6a9bea1f0"}
  ```

## Export

Close the editor first, so that the last edits are committed, then export. Export runs as a
task.

```bash
alcedo-cli call editor.close
alcedo-cli call export.run '{"element_ids":[1,2,3],"out_dir":"D:/Photos/export","format":"JPEG","quality":90,"long_edge":2048}'
alcedo-cli call tasks.wait '{"idle":true}'
alcedo-cli call export.status
```

```json
{"queued":3,"skipped":0,"task_id":"export-1"}
{"idle":true}
{"counts":{"completed":3,"failed":0,"skipped":0,"succeeded":3,"total":3},"error_summary":"","in_flight":false,"items":[{"element_id":1,"image_id":1,"state":"succeeded"},...],"message":"Export complete. Written 3/3 image(s), failed 0.","queue":[],"task_id":"export-1"}
```

The files are `D:/Photos/export/DSC0027.jpg` and so on. A 16-bit TIFF with ZIP compression, in
a subfolder, with a file name pattern:

```bash
alcedo-cli call export.run '{"element_ids":[1],"out_dir":"D:/Photos/export","subfolder":"tiff","format":"TIFF","bit_depth":16,"tiff_compression":"ZIP","file_name_pattern":"{source}-16bit"}'
```

This writes `D:/Photos/export/tiff/DSC0027-16bit.tif`. `export.options` lists the formats and
their bit depths, and the [export reference](reference/export.md) every option.

## Search

```bash
alcedo-cli call search.query '{"text":"sony","limit":5}'
```

```json
{"has_more":false,"limit":5,"matches":[{"camera_model":"ILCE-7M3","capture_date":"2022-09-18","element_id":1,"file_name":"DSC0027.ARW","image_id":1,"lens":"FE 28mm F2","rating":4},...],"offset":0,"route":"traditional","total":3}
```

The search looks at the file name, the camera, the lens, and the other metadata, and at the AI
descriptions when the project has them.

## Retouch every photo

Do the edit loop for each photo of `library.list`. A Bash example that opens each photo and
writes a preview; the agent then looks at each preview and sets the values:

```bash
for id in $(alcedo-cli --json call library.list | python -c "import sys,json; print(' '.join(str(i['element_id']) for i in json.load(sys.stdin)['items']))"); do
  alcedo-cli call editor.open "{\"element_id\":$id}"
  alcedo-cli call render.preview "{\"out_dir\":\"D:/Photos/previews\",\"compare\":\"none\",\"long_edge\":1024,\"file_stem\":\"photo-$id\"}"
done
alcedo-cli call editor.close
```

`library.list` returns 100 photos by default (`limit`, at most 1000); use `offset` for the next
page. `editor.open` of the next photo saves the photo that is open.

## Stop the session

```bash
alcedo-cli call tasks.wait '{"idle":true}'
alcedo-cli session stop
```

```text
session 'trip' stopped
```

`session stop` saves the project before the process exits.

## Edit in the Alcedo Studio window

The Alcedo Studio window serves the session `gui` (see
[The GUI session setting](installation.md#the-gui-session-setting)). The user and the agent take turns:

1. The agent takes control. The window shows an **Agent** task:

   ```bash
   alcedo-cli --session gui call control.acquire '{"holder_name":"claude-code"}'
   ```

   ```json
   {"held":true,"holder_name":"claude-code","revoked_by_user":false}
   ```

2. The agent edits the photo that the user selected (`library.selection.get`) with the same
   commands as above. The user sees each change in the window.

3. The user can click **Take control** at any time. The next agent command fails with `-32004`:
   stop and wait until the user clicks **Return control**, then call `control.acquire` again.

4. The agent gives control back when it is done:

   ```bash
   alcedo-cli --session gui call control.release
   ```

Do not call `session stop` on the `gui` session: it closes the application.
