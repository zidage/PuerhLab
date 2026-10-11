# Task commands

See [Concepts](../concepts.md#tasks) for the task ids and the task object.

## `tasks.list`

<!-- BEGIN GENERATED: tasks.list -->
Changes state: no.

Returns the tasks that have not finished (imports, an export, and background tasks) and whether all application work is idle.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `idle` | boolean | yes |  |
| `tasks` | array of object | yes |  |
| `tasks[].counts` | object | no |  |
| `tasks[].kind` | string | yes |  |
| `tasks[].message` | string | no |  |
| `tasks[].progress` | integer | no |  |
| `tasks[].state` | string | yes |  |
| `tasks[].task_id` | string | yes |  |
<!-- END GENERATED: tasks.list -->

**Errors:** none.

**Example**

```bash
alcedo-cli --json call tasks.list
```

```json
{"idle":true,"tasks":[]}
```

## `tasks.wait`

<!-- BEGIN GENERATED: tasks.wait -->
Changes state: no.

Waits until the task 'task_id' finishes and returns its final state, or with 'idle': true until no import, export, or background task runs.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `idle` | boolean | no |  | Wait for all application work instead. |
| `task_id` | string | no |  | A task id from a command. |

**Result**

The schema lists no fields; see the description and the example.
<!-- END GENERATED: tasks.wait -->

A task that finished before the call answers at once with its final state. A finished task can
end as `failed`; check `state` and `counts`, not only the exit code.

**Errors**

- `-32602`: neither or both of `task_id` and `idle`; or an unknown task id.
- `-32006`: the task did not finish within `timeout_ms`. It still runs. Call `tasks.wait`
  again, with a larger `--timeout` for a large import or export.

**Example**

```bash
alcedo-cli --json call tasks.wait '{"task_id":"export-1"}'
alcedo-cli --json call tasks.wait '{"idle":true}'
```

```json
{"counts":{"completed":2,"failed":0,"skipped":0,"succeeded":2,"total":2},"kind":"export","message":"Export complete. Written 2/2 image(s), failed 0.","progress":100,"state":"succeeded","task_id":"export-1"}
{"idle":true}
```
