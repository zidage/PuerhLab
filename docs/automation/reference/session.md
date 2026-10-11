# Session commands

## `session.ping`

<!-- BEGIN GENERATED: session.ping -->
Changes state: no.

Checks that the session answers.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `pong` | boolean | yes |  |
<!-- END GENERATED: session.ping -->

**Errors:** none.

**Example**

```bash
alcedo-cli --json call session.ping
```

```json
{"pong":true}
```

## `session.describe`

<!-- BEGIN GENERATED: session.describe -->
Changes state: no.

Returns the protocol version, the host mode, the application version, and the command list with the parameter and result schemas.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `application_version` | string | yes |  |
| `commands` | array | yes |  |
| `host_mode` | `headless` \| `gui` | yes |  |
| `protocol_version` | integer | yes |  |
<!-- END GENERATED: session.describe -->

**Errors:** none.

**Example**

```bash
alcedo-cli --json call session.describe
```

```json
{"application_version":"0.3.2","commands":[{"changes_state":false,"description":"Takes control of a GUI session so that state-changing commands run. ...","method":"control.acquire","params_schema":{...},"result_schema":{...}}, ...],"host_mode":"headless","protocol_version":1}
```

`alcedo-cli --json schema` prints only the `commands` array.

## `state.get`

<!-- BEGIN GENERATED: state.get -->
Changes state: no.

Returns the project, workspace, editor session, control state, running tasks, and whether all application work is idle.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `control` | object | yes |  |
| `editor` | object | yes |  |
| `host_mode` | string | yes |  |
| `idle` | boolean | yes |  |
| `project` | object | yes |  |
| `tasks` | array | yes |  |
| `workspace` | string | yes |  |
<!-- END GENERATED: state.get -->

`tasks` holds the tasks that the application keeps, also finished ones; `tasks.list` holds only
the tasks that have not finished. `project.message` is the last status message of the project.

**Errors:** none.

**Example**

```bash
alcedo-cli --json call state.get
```

```json
{"control":{"held":true,"holder_name":"","revoked_by_user":false},"editor":{"element_id":0,"image_id":0,"state":"NoImage"},"host_mode":"headless","idle":true,"project":{"entered":true,"loaded":true,"loading":false,"message":"OpenCL acceleration is ready.","path":"D:/Photos/demo.alcd"},"tasks":[],"workspace":"library"}
```

## `session.shutdown`

<!-- BEGIN GENERATED: session.shutdown -->
Changes state: yes. In a GUI session it needs agent control (`control.acquire`).

Shuts the application modules down, which persists the open project, and then ends the session process. The response arrives after the shutdown.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `persist` | `true` | no | `true` | Persist the open project before the exit. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `shut_down` | boolean | yes |  |
<!-- END GENERATED: session.shutdown -->

`alcedo-cli session stop` sends this command and waits for the process to exit. In a GUI session
the command needs agent control and closes the application.

**Errors:**

- `-32602` `/persist`: `persist` can only be `true`. To close without saving, use
  `project.close` with `{"persist":false}` first.
- `-32003` and `-32004` in a GUI session.

**Example**

```bash
alcedo-cli --json call session.shutdown
```

```json
{"shut_down":true}
```
