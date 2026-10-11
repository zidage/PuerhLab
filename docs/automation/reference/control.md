# Control commands

Agent control applies to the GUI host. See [Concepts](../concepts.md#agent-control-gui-host). In
the headless host these commands always report `held: true` and change nothing.

## `control.acquire`

<!-- BEGIN GENERATED: control.acquire -->
Changes state: no.

Takes control of a GUI session so that state-changing commands run. While the agent holds control, the user cannot change the session state; the task island shows the agent task with the running command. Waits while the user drags an editor control. Fails with -32004 after the user took control until the user selects Return control, and with -32005 while another holder holds control. The headless host always reports held.

**Parameters**

| Name | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `holder_name` | string | no | `"agent"` | The name that the task island shows, for example claude-code. |

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `held` | boolean | yes |  |
| `holder_name` | string | yes |  |
| `revoked_by_user` | boolean | yes |  |
<!-- END GENERATED: control.acquire -->

**Errors**

- `-32004`: the user selected **Take control** and has not selected **Return control** yet.
- `-32005`: another holder has control. `data.holder_name` names it.
- `-32006`: the user kept an editor slider down longer than `timeout_ms`.

**Example** (GUI session)

```bash
alcedo-cli --session gui --json call control.acquire '{"holder_name":"claude-code"}'
```

```json
{"held":true,"holder_name":"claude-code","revoked_by_user":false}
```

After **Take control**:

```json
{"error":{"code":-32004,"data":{"reason":"The user took control. Wait until the user selects Return control in the task dialog."},"message":"The user took control. Wait until the user selects Return control in the task dialog."}}
```

## `control.release`

<!-- BEGIN GENERATED: control.release -->
Changes state: no.

Gives control of a GUI session back to the user. The agent task finishes as succeeded. Without control the state does not change.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `held` | boolean | yes |  |
| `holder_name` | string | yes |  |
| `revoked_by_user` | boolean | yes |  |
<!-- END GENERATED: control.release -->

**Errors:** none.

**Example** (GUI session)

```bash
alcedo-cli --session gui --json call control.release
```

```json
{"held":false,"holder_name":"","revoked_by_user":false}
```

## `control.get`

<!-- BEGIN GENERATED: control.get -->
Changes state: no.

Returns the agent control state of the session.

**Parameters**

No parameters other than `timeout_ms`.

**Result**

| Field | Type | Always | Notes |
| --- | --- | --- | --- |
| `held` | boolean | yes |  |
| `holder_name` | string | yes |  |
| `revoked_by_user` | boolean | yes |  |
<!-- END GENERATED: control.get -->

**Errors:** none.

**Example** (headless session)

```bash
alcedo-cli --json call control.get
```

```json
{"held":true,"holder_name":"","revoked_by_user":false}
```

## A command without control

In a GUI session, a state-changing command without control fails:

```bash
alcedo-cli --session gui --json call editor.set '{"field":"exposure","value":2.5}'
```

```json
{"error":{"code":-32003,"data":{"reason":"This command changes the session state and needs agent control. Call control.acquire first."},"message":"This command changes the session state and needs agent control. Call control.acquire first."}}
```
