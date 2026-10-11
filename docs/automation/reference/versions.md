# Version commands

See [Concepts](../concepts.md#history-and-versions). All Version commands work on the open
photo.

## `versions.list`

<!-- BEGIN GENERATED: versions.list -->
Changes state: no.

Returns the Versions of the open image and the active one, like the Versions panel. created_at and updated_at are Unix seconds.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `active_version_id` | string | yes |  |
| `versions` | array of object | yes |  |
| `versions[].active` | boolean | yes |  |
| `versions[].created_at` | integer | no |  |
| `versions[].head_commit` | string or null | yes | Hex hash of the head commit; null at the root. |
| `versions[].name` | string | yes |  |
| `versions[].updated_at` | integer | no |  |
| `versions[].version_id` | string | yes |  |
<!-- END GENERATED: versions.list -->

`created_at` and `updated_at` are Unix times in seconds.

**Errors**

- `-32001`: no photo is open.

**Example**

```bash
alcedo-cli --json call versions.list
```

```json
{"active_version_id":"320def851a60e0cbb9c03f554d2eb433","versions":[{"active":false,"created_at":1791682754,"head_commit":"2a79d1a0400943005d22a1e89d4b609d","name":"Default","updated_at":1791682757,"version_id":"8d3c296439fbf8e13a5a19993ea08afa"},{"active":true,"created_at":1791682758,"head_commit":"00d40dcc8ff02d41a515617956d72aa7","name":"Black and white","updated_at":1791682759,"version_id":"320def851a60e0cbb9c03f554d2eb433"}]}
```

## `versions.create`

<!-- BEGIN GENERATED: versions.create -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Creates a Version at the image root and checks it out, like New Version in the Versions panel. The open edits are saved first. Without 'name' the Version gets the next default name. Answers after the frame of the new Version.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `name` | string | no |  | The Version name. Leading and trailing white space is removed; an empty name is rejected. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `head_commit` | string or null | yes | Hex hash of the head commit; null at the root. |
| `name` | string | no |  |
| `version_id` | string | yes |  |
<!-- END GENERATED: versions.create -->

The new Version starts at the photo as imported, with the default look and no commits.

**Errors**

- `-32001`: no photo is open.
- `-32602` `/name`: the name is empty after trimming.

**Example**

```bash
alcedo-cli --json call versions.create '{"name":"Black and white"}'
```

```json
{"head_commit":null,"name":"Black and white","version_id":"320def851a60e0cbb9c03f554d2eb433"}
```

## `versions.branch`

<!-- BEGIN GENERATED: versions.branch -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Creates a Version that starts at a commit and checks it out, like Branch in the Versions panel. 'from_commit' is a commit hash (editor.history); without it the head of the active Version. Without 'name' the Version gets the next default name. Answers after the frame of the new Version.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `from_commit` | string | no |  | Hex hash of the start commit. |
| `name` | string | no |  | The Version name. Leading and trailing white space is removed; an empty name is rejected. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `head_commit` | string or null | yes | Hex hash of the head commit; null at the root. |
| `name` | string | no |  |
| `version_id` | string | yes |  |
<!-- END GENERATED: versions.branch -->

**Errors**

- `-32001`: no photo is open.
- `-32602` `/from_commit`: the hash is empty or is not a commit of the active Version; or
  `from_commit` is missing and the active Version has no commit.
- `-32602` `/name`: the name is empty after trimming.

**Example**

```bash
alcedo-cli --json call versions.branch '{"name":"Exposure only","from_commit":"ba09cf040425da711e4cb9b431effa64"}'
```

```json
{"head_commit":"ba09cf040425da711e4cb9b431effa64","name":"Exposure only","version_id":"44f0ca29ddcc81dfb8f0f02204d3ea19"}
```

## `versions.checkout`

<!-- BEGIN GENERATED: versions.checkout -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Checks out a Version, like a click on its card in the Versions panel. The open edits are saved first. Answers after the frame of the Version head.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `version_id` | string | yes |  | Hex id of a Version (versions.list). |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `head_commit` | string or null | yes | Hex hash of the head commit; null at the root. |
| `name` | string | no |  |
| `version_id` | string | yes |  |
<!-- END GENERATED: versions.checkout -->

**Errors**

- `-32001`: no photo is open.
- `-32602` `/version_id`: unknown Version id.

**Example**

```bash
alcedo-cli --json call versions.checkout '{"version_id":"8d3c296439fbf8e13a5a19993ea08afa"}'
```

```json
{"head_commit":"2a79d1a0400943005d22a1e89d4b609d","name":"Default","version_id":"8d3c296439fbf8e13a5a19993ea08afa"}
```

## `versions.rename`

<!-- BEGIN GENERATED: versions.rename -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Renames a Version, like Rename in the Versions panel. A rename to the current name changes nothing and succeeds.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `name` | string | yes |  | The Version name. Leading and trailing white space is removed; an empty name is rejected. |
| `version_id` | string | yes |  | Hex id of a Version (versions.list). |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `name` | string | yes |  |
| `version_id` | string | yes |  |
<!-- END GENERATED: versions.rename -->

**Errors**

- `-32001`: no photo is open.
- `-32602` `/version_id`: unknown Version id.
- `-32602` `/name`: the name is empty after trimming.

**Example**

```bash
alcedo-cli --json call versions.rename '{"version_id":"44f0ca29ddcc81dfb8f0f02204d3ea19","name":"Exposure 0.5"}'
```

```json
{"name":"Exposure 0.5","version_id":"44f0ca29ddcc81dfb8f0f02204d3ea19"}
```

## `versions.remove`

<!-- BEGIN GENERATED: versions.remove -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Removes a Version, like Remove in the Versions panel.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `version_id` | string | yes |  | Hex id of a Version (versions.list). |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `active_version_id` | string | yes |  |
| `version_id` | string | yes |  |
<!-- END GENERATED: versions.remove -->

**Errors**

- `-32001`: no photo is open.
- `-32602` `/version_id`: unknown Version id.
- `-32002`: `The active Version or the final remaining Version cannot be removed`. Check out
  another Version first.

**Example**

```bash
alcedo-cli --json call versions.remove '{"version_id":"320def851a60e0cbb9c03f554d2eb433"}'
```

```json
{"active_version_id":"44f0ca29ddcc81dfb8f0f02204d3ea19","version_id":"320def851a60e0cbb9c03f554d2eb433"}
```
