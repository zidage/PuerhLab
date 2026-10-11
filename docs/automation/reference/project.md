# Project commands

A project is one packed file (`.alcd`). A session has at most one project open. The headless
host opens or creates its project at start; these commands switch it.

## `project.create`

<!-- BEGIN GENERATED: project.create -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Creates a packed project named 'name' in 'folder' and enters it. A loaded project closes first and is saved. The response arrives when the new project is loaded.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `folder` | string | yes |  | An existing folder. |
| `name` | string | yes |  | The project name. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `entered` | boolean | yes |  |
| `name` | string | yes |  |
| `path` | string | yes |  |
| `photo_count` | integer | yes |  |
<!-- END GENERATED: project.create -->

**Errors**

- `-32602` `/folder`: the folder does not exist.
- `-32602` `/name`: `the project name is empty`.
- `-32005`: another project switch is queued.

**Example**

```bash
alcedo-cli --json call project.create '{"folder":"D:/Photos/newproject","name":"second"}'
```

```json
{"entered":true,"name":"second","path":"D:/Photos/newproject/second.alcd","photo_count":0}
```

## `project.open`

<!-- BEGIN GENERATED: project.open -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Opens the packed project at 'path' and enters it. A loaded project closes first and is saved. The response arrives when the project is loaded.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `path` | string | yes |  | The .alcd project file. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `entered` | boolean | yes |  |
| `name` | string | yes |  |
| `path` | string | yes |  |
| `photo_count` | integer | yes |  |
<!-- END GENERATED: project.open -->

**Errors**

- `-32602` `/path`: the project file does not exist.
- `-32007`: the project cannot be loaded; `data.reason` tells why.

**Example**

```bash
alcedo-cli --json call project.open '{"path":"D:/Photos/demo.alcd"}'
```

```json
{"entered":true,"name":"demo","path":"D:/Photos/demo.alcd","photo_count":3}
```

## `project.save`

<!-- BEGIN GENERATED: project.save -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Saves and packs the loaded project. Like File > Save Project, it first closes the editor image with its changes.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `path` | string | yes |  |
<!-- END GENERATED: project.save -->

`project.save` closes the open editor image. Call `editor.open` again to go on editing.

**Errors**

- `-32001`: `No project is loaded.`
- `-32007`: the save failed.

**Example**

```bash
alcedo-cli --json call project.save
```

```json
{"path":"D:/Photos/demo.alcd"}
```

## `project.close`

<!-- BEGIN GENERATED: project.close -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Closes the editor image and the loaded project. With persist (the default), both are saved first; without it, the changes since the last save are discarded. The response arrives when no project is loaded.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `persist` | boolean | no | `true` | Save the editor image and the project first. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `closed` | boolean | yes |  |
<!-- END GENERATED: project.close -->

**Errors**

- `-32007`: the editor cannot close its image, for example after a failed editor save.

**Example**

```bash
alcedo-cli --json call project.close
```

```json
{"closed":true}
```
