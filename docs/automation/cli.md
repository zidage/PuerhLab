# Command-line interface

```text
alcedo-cli [global options] <command> [arguments]
```

## Global options

Global options can stand at any position.

| Option | Meaning |
| --- | --- |
| `--session <name>` | Use this session. Default: the environment variable `ALCEDO_SESSION`, then the only live session. `session start` uses `default`. |
| `--session-dir <dir>` | The directory of the session files. Default: see [Concepts](concepts.md#session-directory-and-session-files). |
| `--json` | Print compact JSON on one line: the result object, or `{"error": {...}}`. Use it in scripts and agents. |
| `--timeout <ms>` | Time limit of the command. `call` sends it as `timeout_ms` when the parameters have none. Default 120000. |

## Commands

### call

```text
alcedo-cli call <method> [<params-json>]
```

Sends one command and prints its response. `<params-json>` is one JSON object; leave it out for
no parameters. Every command also accepts `timeout_ms` (default 120000): the time limit for the
command to reach its terminal state.

```bash
alcedo-cli --json call library.list '{"limit":2}'
```

```json
{"items":[{"element_id":1,"file_name":"DSC0027.ARW","image_id":1,"rating":0},{"element_id":2,"file_name":"DSC00800.ARW","image_id":2,"rating":0}],"offset":0,"total":3}
```

Without `--json` the result is printed as indented JSON, and an error is printed to stderr as
`error <code> <name>: <message>` followed by `data`.

### schema

```text
alcedo-cli schema
```

Prints the command list: the method, `(changes state)` for a command that changes state, and
the description. With `--json` it prints the full `commands` array of `session.describe`, with
the parameter and result schemas.

### session start

```text
alcedo-cli session start --headless (--project <path> | --create <folder>,<name>)
    [--editor-backend cuda|opencl|metal] [--viewport <w>x<h>] [--settings-dir <dir>]
    [--log-file <path>] [--host-binary <path>]
```

Starts `alcedo_main --headless` and waits until its session answers `session.ping`. The host
keeps running after `alcedo-cli` exits.

| Option | Meaning |
| --- | --- |
| `--project <path>` | Open this packed project (`.alcd`). |
| `--create <folder>,<name>` | Create the project `<folder>/<name>.alcd` and open it. The folder must exist. |
| `--editor-backend` | The GPU backend of the editor. Windows default: `opencl`. macOS: `metal`. |
| `--viewport <w>x<h>` | The size of the off-screen editor viewport. Default `1920x1080`. |
| `--settings-dir <dir>` | Keep the application settings in this folder instead of the user settings. |
| `--log-file <path>` | Write the host log to this file. |
| `--host-binary <path>` | The host program. Default: see [Installation](installation.md#where-the-programs-are). |

The session name is `--session`, or `default`. When a live session already has the name, the
command fails with exit code 2. When the host exits before it is ready, the command prints the
end of its error output (kept in `<session-dir>/<name>.start.log`) and fails with exit code 3.

```bash
alcedo-cli --session docs session start --headless --create D:/Photos,demo
```

```text
session 'docs' started (process 15480, project D:/Photos/demo.alcd)
```

With `--json`:

```json
{"host_mode":"headless","pid":23420,"project_path":"D:/Photos/demo.alcd","session_file":"C:/Users/me/AppData/Local/Alcedo/Alcedo/automation/sessions/docs.json","session_name":"docs","socket_name":"alcedo-automation-23420-docs"}
```

### session stop

Sends `session.shutdown`, which saves the open project, and waits up to 60 seconds for the
process to exit.

```text
session 'docs' stopped
```

A GUI session cannot be stopped from the CLI: `session.shutdown` changes state and needs agent
control, and it closes the window. Quit the application instead.

### session list

Prints one line for each session file: name, state (`live`, `stale`, `unsupported-protocol`,
or `unreadable`), host mode, process id, and project path.

```text
docs	live	headless	pid 15480	D:/Photos/demo.alcd
gui	live	gui	pid 36984	D:/Photos/trip.alcd
```

### session prune

Removes the stale session files and prints their paths.

### watch

Prints the notifications of the session until you stop it (Ctrl+C), one per line:
`<method> <params>`. With `--json`, each line is the JSON-RPC notification.

```text
control.changed {"held":true,"holder_name":"claude-code","revoked_by_user":false}
history.changed {"element_id":41,"head_commit":"90169a9b3c61d41fb1df19a469dc07fe","image_id":41,"version_id":"21602a753ff9b98f08516e0a9a7bf78d"}
```

## Exit codes

| Code | Meaning | Examples |
| --- | --- | --- |
| 0 | Success. | |
| 1 | The command answered with an error. The error code tells why. | `-32602` for a bad parameter. |
| 2 | Session selection failed. | No live session; several live sessions and no `--session`; the name does not exist; the process of the session does not run; `session start` with a name that is live. |
| 3 | Connection failed. | The socket does not answer; the host exited before it was ready; the response did not arrive in time. |
| 4 | Usage error. | Unknown command or option; `<params-json>` is not a JSON object. |

## Error codes

An error response has `code`, `message`, and often `data`. `data.reason` repeats the message of
the owner. `data.pointer` points at the parameter that is wrong, for example `/value`.

| Code | Name | Meaning | What to do |
| --- | --- | --- | --- |
| -32700 | parse error | The line is not JSON. | Fix the request. The CLI never sends this. |
| -32600 | invalid request | The message is not a JSON-RPC request. | Fix the request. |
| -32601 | method not found | Unknown command. | Check the name with `alcedo-cli schema`. |
| -32602 | invalid params | A parameter is missing, unknown, of the wrong type, or out of range. | Read `data.pointer` and `data.reason`. Ranges are in the schema and in `editor.catalog`. |
| -32603 | internal error | A defect in the host. | Report it with the host log. |
| -32001 | not ready | The state that the command needs does not exist: no project, no editor image, no library selection in the headless host. | Open a project or a photo first. |
| -32002 | rejected | The owner refused the operation. For example: the active Version cannot be removed. | Read `data.reason`. |
| -32003 | control not held | A state-changing command in a GUI session without agent control. | Call `control.acquire`. |
| -32004 | control revoked by user | The user selected **Take control**. | Stop. Wait until the user selects **Return control**, then call `control.acquire`. |
| -32005 | busy | A resource is in use: another holder has control, or the editor renders another job (for example an open comparison). | Wait and try again. |
| -32006 | timeout | The command did not reach its terminal state within `timeout_ms`. | The operation can still finish. Check with `state.get` or `editor.history` before you repeat it. |
| -32007 | failed | The operation started and failed. | Read `data.reason`. |

```json
{"error":{"code":-32602,"data":{"pointer":"/value","reason":"exposure must be in [-10, 10]"},"message":"exposure must be in [-10, 10]"}}
```

## Protocol

The CLI is one client of the protocol. Other programs can connect to the socket of the session
file and send the same messages.

- Each message is one UTF-8 JSON object on one line, ended by `\n`.
- Requests and responses follow JSON-RPC 2.0: `{"jsonrpc":"2.0","id":1,"method":"session.ping","params":{}}`.
- Notifications from the host have no `id`.
- `session.describe` returns `protocol_version` (now 1). The CLI refuses a session with another
  protocol version.
