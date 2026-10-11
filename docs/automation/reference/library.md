# Library commands

## `library.import`

<!-- BEGIN GENERATED: library.import -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Imports the files 'paths', or the supported files of 'folder' (with its subfolders when 'recursive', the default), into the current folder of the open project. Returns a task id for tasks.wait as soon as the import starts.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `folder` | string | no |  | A folder to import. |
| `paths` | array of string | no |  | at least 1 item(s). Files to import. |
| `recursive` | boolean | no | `true` | Include the subfolders of 'folder'. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `task_id` | string | yes |  |
<!-- END GENERATED: library.import -->

Give `folder` or `paths`. Wait for the task before you list the photos: `tasks.wait` with the
`task_id`, or with `idle: true`.

**Errors**

- `-32602`: `give exactly one of paths and folder`.
- `-32602` `/folder` or `/paths/<n>`: the folder or file does not exist.
- `-32001`: `No project is open.`
- `-32005`: `An import is already running.` Wait with `tasks.wait`, then try again.

**Example**

```bash
alcedo-cli --json call library.import '{"folder":"D:/Photos/2026-trip"}'
alcedo-cli --json call tasks.wait '{"task_id":"import-1"}'
```

```json
{"task_id":"import-1"}
{"counts":{"completed":1,"excluded":0,"failed":0,"total":1,"unsupported":0},"kind":"import","message":"Import complete: 1 imported, 0 not selected, 0 unsupported, 0 failed","progress":100,"state":"succeeded","task_id":"import-1"}
```

## `library.folders`

<!-- BEGIN GENERATED: library.folders -->
Changes state: no.

Returns the folder tree of the open project as a list, parents before children. The root folder is first.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `folders` | array | yes |  |
<!-- END GENERATED: library.folders -->

**Errors**

- `-32001`: no project is loaded.

**Example**

```bash
alcedo-cli --json call library.folders
```

```json
{"folders":[{"depth":0,"folder_id":0,"name":"","parent_id":0,"path":"/"}]}
```

## `library.list`

<!-- BEGIN GENERATED: library.list -->
Changes state: no.

Returns one page of the photos of a folder (the root folder by default) in the order of 'sort', then by element id. 'total' is the photo count of the folder.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `descending` | boolean | no | `false` |  |
| `folder_id` | integer | no |  | at least 0. A folder id from library.folders. |
| `limit` | integer | no | `100` | 1 to 1000. |
| `offset` | integer | no | `0` | at least 0. |
| `sort` | `none` \| `capture_time` \| `import_time` \| `camera_model` \| `lens` \| `rating` \| `labels` \| `edit_time` | no | `"none"` |  |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `items` | array of object | yes |  |
| `items[].element_id` | integer | yes |  |
| `items[].file_name` | string | yes |  |
| `items[].image_id` | integer | yes |  |
| `items[].rating` | integer | yes |  |
| `offset` | integer | yes |  |
| `total` | integer | yes |  |
<!-- END GENERATED: library.list -->

An unknown `folder_id` gives an empty page. To read all photos, repeat with `offset` until
`offset` + the item count reaches `total`.

**Errors**

- `-32001`: no project is loaded.

**Example**

```bash
alcedo-cli --json call library.list '{"limit":10}'
```

```json
{"items":[{"element_id":1,"file_name":"DSC0027.ARW","image_id":1,"rating":0},{"element_id":2,"file_name":"DSC00800.ARW","image_id":2,"rating":0},{"element_id":3,"file_name":"DSC06683.dng","image_id":3,"rating":0}],"offset":0,"total":3}
```

## `library.thumbnail`

<!-- BEGIN GENERATED: library.thumbnail -->
Changes state: no.

Renders the library thumbnail of a photo and writes it as a PNG file whose long edge is 'long_edge'.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `element_id` | integer | yes |  | at least 1. |
| `long_edge` | integer | no | `512` | 16 to 2048. |
| `out` | string | yes |  | The PNG file. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `height` | integer | yes |  |
| `path` | string | yes |  |
| `width` | integer | yes |  |
<!-- END GENERATED: library.thumbnail -->

The thumbnail shows the committed edits of the photo. Edits of the open editor image show after
the editor commits them (`editor.close`, `project.save`). The library keeps a thumbnail cache,
so a thumbnail made just after an edit can still show the earlier edits. To see the current
edits of the open photo, use `render.preview`.

**Errors**

- `-32602` `/element_id`: unknown element id.

**Example**

```bash
alcedo-cli --json call library.thumbnail '{"element_id":2,"out":"D:/Photos/thumbs/DSC00800.png","long_edge":256}'
```

```json
{"height":171,"path":"D:/Photos/thumbs/DSC00800.png","width":256}
```

## `library.rate`

<!-- BEGIN GENERATED: library.rate -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Sets the star rating (0 to 5) of 'element_ids'. One photo is rated at once and the result holds 'applied_count'. Several photos use the batch save, and the result holds the 'task_id' of the save task.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `element_ids` | array of integer | yes |  | at least 1 item(s); each item at least 1. |
| `rating` | integer | yes |  | 0 to 5. |

**Result**

The schema lists no fields; see the description and the example.
<!-- END GENERATED: library.rate -->

**Errors**

- `-32602` `/element_ids`: unknown ids; `data.unknown_ids` lists them. Nothing is rated.
- `-32602` `/rating`: the rating is not 0 to 5.

**Example**

```bash
alcedo-cli --json call library.rate '{"element_ids":[3],"rating":2}'
alcedo-cli --json call library.rate '{"element_ids":[1,2],"rating":4}'
```

```json
{"applied_count":1,"rating":2}
{"rated_count":2,"rating":4,"task_id":"bgtask-2"}
```

## `library.delete`

<!-- BEGIN GENERATED: library.delete -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Deletes 'element_ids'. With scope 'project' the photos leave the project; this needs the root folder as the current folder. With scope 'album' they leave the current album.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `element_ids` | array of integer | yes |  | at least 1 item(s); each item at least 1. |
| `scope` | `project` \| `album` | yes |  |  |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `deleted_ids` | array | yes |  |
| `failed_ids` | array | yes |  |
<!-- END GENERATED: library.delete -->

`library.delete` removes the photos from the project or the album. It does not delete the
image files on the disk.

**Errors**

- `-32602` `/element_ids`: unknown ids; `data.unknown_ids` lists them. Nothing is deleted.
- `-32602` `/scope`: `scope` is missing.
- `-32002`: the scope does not fit the current folder, for example `The current folder gives
  the delete scope 'folder', not 'project'.`
- `-32007`: no photo was deleted.

When some photos cannot be deleted, the command succeeds and `failed_ids` lists them.

**Example**

```bash
alcedo-cli --json call library.delete '{"element_ids":[3],"scope":"project"}'
```

```json
{"deleted_ids":[3],"failed_ids":[],"message":"Deleted 1 image(s)."}
```

## `library.selection.get`

<!-- BEGIN GENERATED: library.selection.get -->
Changes state: no.

Returns the library selection of the GUI. The headless host has no selection.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `items` | array | yes |  |
<!-- END GENERATED: library.selection.get -->

**Errors**

- `-32001`: headless host.

**Example** (GUI session)

```bash
alcedo-cli --session gui --json call library.selection.get
```

```json
{"items":[{"element_id":41,"file_name":"DSC02240.ARW","image_id":41}]}
```

Headless:

```json
{"error":{"code":-32001,"data":{"reason":"The library selection exists only in the GUI."},"message":"The library selection exists only in the GUI."}}
```

## `library.selection.set`

<!-- BEGIN GENERATED: library.selection.set -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Replaces the library selection of the GUI with 'element_ids'. The headless host has no selection.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `element_ids` | array of integer | yes |  | each item at least 1. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `items` | array | yes |  |
<!-- END GENERATED: library.selection.set -->

An empty list clears the selection.

**Errors**

- `-32001`: headless host.
- `-32003`, `-32004`: GUI session without agent control.

**Example** (GUI session, with agent control)

```bash
alcedo-cli --session gui --json call library.selection.set '{"element_ids":[41]}'
```

```json
{"items":[{"element_id":41,"file_name":"DSC02240.ARW","image_id":41}]}
```
