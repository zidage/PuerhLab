# Editor commands

The editor edits one photo at a time. `editor.open` opens it; the other editor commands work on
the open photo. Values are UI values; see [Editor fields](editor_fields.md) and
[Concepts](../concepts.md#ui-values).

## `editor.open`

<!-- BEGIN GENERATED: editor.open -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Opens a photo in the editor, like a double-click in the library. An open image is saved and replaced. Answers when the editor can edit the photo.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `element_id` | integer | yes |  | at least 1. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `element_id` | integer | yes |  |
| `head_commit` | string or null | yes | Hex hash of the head commit; null at the root. |
| `image_id` | integer | yes |  |
| `state` | string | yes |  |
<!-- END GENERATED: editor.open -->

**Errors**

- `-32602` `/element_id`: unknown element id.
- `-32001`: `No project is open.`
- `-32007`: the editor is in a failed state (`Failed`, or `RetainedImageFailure` after a failed
  save) and cannot open another photo.

**Example**

```bash
alcedo-cli --json call editor.open '{"element_id":1}'
```

```json
{"element_id":1,"head_commit":null,"image_id":1,"state":"Interactive"}
```

## `editor.close`

<!-- BEGIN GENERATED: editor.close -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Closes the editor image and returns to the library. With persist (the default) the edits are saved first; without it, edits that are not saved are discarded. Answers when the image is closed. A failed save keeps the image open and answers with the reason.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `persist` | boolean | no | `true` |  |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `closed` | boolean | yes |  |
<!-- END GENERATED: editor.close -->

**Errors**

- `-32007`: the save failed, or an earlier save failed and the editor waits for recovery
  ("Resolve the failed editor save before closing the editor").

When no photo is open, `editor.close` succeeds and returns `{"closed":true}`.

**Example**

```bash
alcedo-cli --json call editor.close
```

```json
{"closed":true}
```

## `editor.catalog`

<!-- BEGIN GENERATED: editor.catalog -->
Changes state: no.

Returns every editor field that editor.set accepts, in UI units: the values that the editor panels show. A scalar field has its range (ui_min, ui_max), default, step, and shown decimals. An object field lists its properties (number range, boolean, string, or option list, or a list of numbers) and its default object; a write may give some of the properties. A model field (curve, lut) takes its Model JSON, described in model_shape; a write gives the complete value. A Kelvin property also has its slider position scale (ui_slider), and odt lists the valid encoding_eotf values of each encoding_space. A value outside the range or the option list is rejected; it is not clamped.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `fields` | array of object | yes |  |
| `fields[].aliases` | array of string | no |  |
| `fields[].encoding_eotf_by_space` | object | no |  |
| `fields[].encoding_eotf_by_space.p3_d65` | array of string | yes |  |
| `fields[].encoding_eotf_by_space.rec2020` | array of string | yes |  |
| `fields[].encoding_eotf_by_space.rec709` | array of string | yes |  |
| `fields[].field` | string | yes |  |
| `fields[].kind` | `scalar` \| `object` \| `model` | yes |  |
| `fields[].model_shape` | string | no |  |
| `fields[].panel` | string | yes |  |
| `fields[].properties` | array of object | no |  |
| `fields[].properties[].name` | string | yes |  |
| `fields[].properties[].options` | array of object | no |  |
| `fields[].properties[].options[].label` | string | yes |  |
| `fields[].properties[].options[].value` | string | yes |  |
| `fields[].properties[].type` | `number` \| `boolean` \| `string` \| `option` \| `number_list` | yes |  |
| `fields[].properties[].ui_count` | integer | no |  |
| `fields[].properties[].ui_decimals` | integer | no |  |
| `fields[].properties[].ui_max` | number | no |  |
| `fields[].properties[].ui_min` | number | no |  |
| `fields[].properties[].ui_slider` | object | no |  |
| `fields[].properties[].ui_slider.pivot_kelvin` | number | yes |  |
| `fields[].properties[].ui_slider.pivot_position` | integer | yes |  |
| `fields[].properties[].ui_slider.position_max` | integer | yes |  |
| `fields[].properties[].ui_slider.position_min` | integer | yes |  |
| `fields[].properties[].ui_slider.scale` | `kelvin_pivot` | yes |  |
| `fields[].properties[].ui_step` | number | no |  |
| `fields[].ui_decimals` | integer | no |  |
| `fields[].ui_default` | number or object | yes |  |
| `fields[].ui_max` | number | no |  |
| `fields[].ui_min` | number | no |  |
| `fields[].ui_step` | number | no |  |
<!-- END GENERATED: editor.catalog -->

The catalog is the same for every photo and does not need an open photo.
[Editor fields](editor_fields.md) shows it as tables.

**Errors:** none.

**Example**

```bash
alcedo-cli --json call editor.catalog
```

```json
{"fields":[{"field":"exposure","kind":"scalar","panel":"tone","ui_decimals":2,"ui_default":0,"ui_max":10,"ui_min":-10,"ui_step":0.01},{"field":"contrast","kind":"scalar","panel":"tone","ui_decimals":0,"ui_default":0,"ui_max":100,"ui_min":-100,"ui_step":1}, ...]}
```

## `editor.get`

<!-- BEGIN GENERATED: editor.get -->
Changes state: no.

Returns the UI values of the open image: the values that the editor panels show, in the units of editor.catalog. Without 'fields', every field that the image shows.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `fields` | array of string | no |  |  |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `element_id` | integer | yes |  |
| `head_commit` | string or null | yes | Hex hash of the head commit; null at the root. |
| `image_id` | integer | yes |  |
| `state` | string | yes |  |
| `values` | object | yes |  |
<!-- END GENERATED: editor.get -->

**Errors**

- `-32001`: no photo is open.
- `-32602` `/fields/<n>`: unknown field.

**Example**

```bash
alcedo-cli --json call editor.get '{"fields":["exposure","color_temp","crop_rotate"]}'
```

```json
{"element_id":1,"head_commit":null,"image_id":1,"state":"Interactive","values":{"color_temp":{"kelvin":5121.59033203125,"mode":"as_shot","tint":14.783473014831543},"crop_rotate":{"angle_degrees":0,"aspect_height":1,"aspect_preset":"free","aspect_width":1,"height":1,"width":1,"x":0,"y":0},"exposure":1.5}}
```

With `mode` `as_shot`, `color_temp` reports the white balance of the camera.

## `editor.set`

<!-- BEGIN GENERATED: editor.set -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Sets one field of the open image to a UI value, like releasing a panel control: one history commit. Answers after the commit and its rendered frame, with the value that the editor now shows (a crop can be constrained). A value equal to the current one makes no commit. The commit has the author agent and the optional 'note'.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `field` | string | yes |  |  |
| `note` | string | no |  | A note for the commits of this call. The History panel shows it with the author agent. |
| `value` | number or object | yes |  | The UI value: a number for a scalar field, an object for an object or Model field. editor.catalog gives the range, the properties, and the Model shape. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `field` | string | yes |  |
| `head_commit` | string or null | yes | Hex hash of the head commit; null at the root. |
| `value` | number or object or null | yes |  |
<!-- END GENERATED: editor.set -->

**Errors**

- `-32602` `/field`: unknown field. `editor.catalog` lists the fields.
- `-32602` `/value`: the value is out of range, has the wrong type, or names an unknown
  property or option. `data.reason` gives the range, for example
  `exposure must be in [-10, 10]`.
- `-32002` `Adjustments require an interactive image`: no photo is open, or the editor is
  loading, saving, or in a failed state. `state` in the result of `editor.get` tells which.

**Example**

```bash
alcedo-cli --json call editor.set '{"field":"exposure","value":0.5,"note":"darker"}'
```

```json
{"field":"exposure","head_commit":"ba09cf040425da711e4cb9b431effa64","value":0.5}
```

An object field takes some of its properties:

```bash
alcedo-cli --json call editor.set '{"field":"color_temp","value":{"mode":"custom","kelvin":5200,"tint":4}}'
```

## `editor.batch_set`

<!-- BEGIN GENERATED: editor.batch_set -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Sets several fields in order. Each change that alters a value is one history commit, applied after the frame of the previous one. The first change that fails stops the batch: the error data has 'applied' (the changes before it, which stay) and 'head_commit'. Each commit has the author agent and the optional 'note'.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `changes` | array of object | yes |  | at least 1 item(s). |
| `changes[].field` | string | yes |  |  |
| `changes[].value` | number or object | yes |  | The UI value: a number for a scalar field, an object for an object or Model field. editor.catalog gives the range, the properties, and the Model shape. |
| `note` | string | no |  | A note for the commits of this call. The History panel shows it with the author agent. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `applied` | integer | yes |  |
| `head_commit` | string or null | yes | Hex hash of the head commit; null at the root. |
| `values` | object | yes |  |
<!-- END GENERATED: editor.batch_set -->

All changes are checked before the first one is applied. In a GUI session, control is checked
again before each change after the first.

**Errors**

- `-32602` `/changes`: `a batch has at most 64 changes`. Nothing is applied.
- `-32602` `/changes/<n>/field` or `/changes/<n>/value`: change `<n>` is not valid, as for
  `editor.set`.
- `-32002` `Adjustments require an interactive image`: as for `editor.set`.
- `-32004`: the user took control.

The changes apply in order and each one makes a commit. When a change fails, the changes before
it stay: `data.applied` is their number and `data.head_commit` the head after them, for example
`{"applied":1,"head_commit":"9b8b...","pointer":"/changes/1/value","reason":"contrast must be in
[-100, 100]"}`. Undo them with `editor.undo` if you want all or nothing.

**Example**

```bash
alcedo-cli --json call editor.batch_set '{"changes":[{"field":"contrast","value":10},{"field":"color_temp","value":{"mode":"custom","kelvin":5200,"tint":4}}],"note":"tone and white balance"}'
```

```json
{"applied":2,"head_commit":"2a79d1a0400943005d22a1e89d4b609d","values":{"color_temp":{"kelvin":5200,"mode":"custom","tint":4},"contrast":10}}
```

## `editor.undo`

<!-- BEGIN GENERATED: editor.undo -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Moves the history head back one commit, like Edit > Undo. Answers after the frame of the restored values.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `can_redo` | boolean | yes |  |
| `can_undo` | boolean | yes |  |
| `head_commit` | string or null | yes | Hex hash of the head commit; null at the root. |
<!-- END GENERATED: editor.undo -->

**Errors**

- `-32001`: no photo is open.
- `-32002`: `Nothing to undo`.

**Example**

```bash
alcedo-cli --json call editor.undo
```

```json
{"can_redo":true,"can_undo":true,"head_commit":"e8b2f2f8e394c0d4eea9cbb0837eb1be"}
```

## `editor.redo`

<!-- BEGIN GENERATED: editor.redo -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Moves the history head forward one commit, like Edit > Redo. Answers after the frame of the restored values.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `can_redo` | boolean | yes |  |
| `can_undo` | boolean | yes |  |
| `head_commit` | string or null | yes | Hex hash of the head commit; null at the root. |
<!-- END GENERATED: editor.redo -->

**Errors**

- `-32001`: no photo is open.
- `-32002`: `Nothing to redo`.

**Example**

```bash
alcedo-cli --json call editor.redo
```

```json
{"can_redo":false,"can_undo":true,"head_commit":"2a79d1a0400943005d22a1e89d4b609d"}
```

## `editor.history`

<!-- BEGIN GENERATED: editor.history -->
Changes state: no.

Returns the history of the active Version of the open image: its commits from the root, the head, and whether undo and redo are possible. 'position' is applied, current (the head), or future (undone, can be redone). 'author' is user or agent; 'note' is the note of an agent commit.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `can_redo` | boolean | yes |  |
| `can_undo` | boolean | yes |  |
| `commits` | array | yes |  |
| `head_commit` | string or null | yes | Hex hash of the head commit; null at the root. |
| `version_id` | string | yes |  |
<!-- END GENERATED: editor.history -->

The list holds every commit of the active Version back to the root, the newest first. `position`
is `current` for the head, `applied` for the commits before it, and `future` for the undone
commits that `editor.redo` can apply again. `field` names the editor field of a commit.

**Errors**

- `-32001`: no photo is open.

**Example**

```bash
alcedo-cli --json call editor.history
```

```json
{"can_redo":false,"can_undo":true,"commits":[{"author":"agent","commit_hash":"2a79d1a0400943005d22a1e89d4b609d","field":"color_temp","note":"tone and white balance","operation_kind":"set_parameter","parent_hash":"e8b2f2f8e394c0d4eea9cbb0837eb1be","position":"current"},{"author":"agent","commit_hash":"e8b2f2f8e394c0d4eea9cbb0837eb1be","field":"contrast","note":"tone and white balance","operation_kind":"set_parameter","parent_hash":"ba09cf040425da711e4cb9b431effa64","position":"applied"},{"author":"agent","commit_hash":"ba09cf040425da711e4cb9b431effa64","field":"exposure","note":"darker","operation_kind":"set_parameter","parent_hash":null,"position":"applied"}],"head_commit":"2a79d1a0400943005d22a1e89d4b609d","version_id":"8d3c296439fbf8e13a5a19993ea08afa"}
```

## `editor.actions`

<!-- BEGIN GENERATED: editor.actions -->
Changes state: no.

Returns which editor actions the session admits now, with the reason for each refused one. The GUI enables its controls from the same decisions.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `actions` | array | yes |  |
<!-- END GENERATED: editor.actions -->

**Errors:** none.

**Example**

```bash
alcedo-cli --json call editor.actions
```

```json
{"actions":[{"allowed":true,"name":"SelectImage"},{"allowed":true,"name":"PreviewAdjustment"},{"allowed":true,"name":"CommitAdjustment"},{"allowed":false,"name":"Undo","reason":"Nothing to undo"},{"allowed":false,"name":"Redo","reason":"Nothing to redo"},{"allowed":true,"name":"MoveHead"}, ...]}
```
