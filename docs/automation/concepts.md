# Concepts

## Session

A session is one running `alcedo_main` process with an automation server. The server listens on
a local socket (a named pipe on Windows, a Unix domain socket on macOS). Only the same OS user
can connect. Several clients can connect at the same time; each connection is independent.

A session has a name. The CLI uses the name to find the socket. Names are 1 to 64 characters of
`A-Z a-z 0-9 . _ -`.

| Host | Name | Started by |
| --- | --- | --- |
| Headless | `default`, or the name of `--session` | `alcedo-cli session start --headless` |
| GUI | `gui`, or `gui-<pid>` when another live process already uses `gui` | The Alcedo Studio window, at start |

## Session directory and session files

The host writes a session file `<name>.json` into the session directory after its server
listens. It removes the file when it shuts down.

| System | Default session directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%\Alcedo\Alcedo\automation\sessions` |
| macOS | `~/Library/Application Support/Alcedo/Alcedo/automation/sessions` |

`--session-dir <dir>` changes the directory for both `alcedo-cli` and the headless host. Use
it to keep test sessions apart from your normal sessions.

```json
{
  "protocol_version": 1,
  "session_name": "docs",
  "socket_name": "alcedo-automation-23420-docs",
  "pid": 23420,
  "host_mode": "headless",
  "project_path": "D:/Photos/demo.alcd",
  "started_at": "2026-10-10T09:00:00Z"
}
```

A session file is **stale** when its process does not run or its socket does not answer
`session.ping`, for example after a crash. `alcedo-cli session list` shows the state of each file
and `alcedo-cli session prune` removes the stale files.

The CLI selects the session in this order:

1. `--session <name>`.
2. The environment variable `ALCEDO_SESSION`.
3. The only live session. When several sessions are live, the CLI stops with exit code 2 and
   lists their names.

## Headless host and GUI host

| | Headless host | GUI host |
| --- | --- | --- |
| Window | None | The Alcedo Studio window |
| Project | Opened or created at start (`--project` or `--create`) | The project that the window has open; can be none |
| Agent control | Always held; `control.*` reports `held: true` | Needed for state-changing commands |
| Library selection | None (`library.selection.*` gives `-32001`) | The selection of the library view |
| Editor rendering | Off-screen, with the editor backend of `--editor-backend` (Windows default: OpenCL) | The window viewport |
| End | `alcedo-cli session stop` or `session.shutdown` | The user quits the application |

`session.describe` and `state.get` report `host_mode`.

## Agent control (GUI host)

In the GUI host the user and an agent share one editor. Agent control takes turns between them.

- Each command has `changes_state` in its schema (`session.describe`). Without control, a command
  that changes state fails with `-32003`. Reads always run.
- `control.acquire` takes control. It waits while the user drags an editor slider, and answers
  when the user releases it. The task island of the window then shows a task of the kind
  **Agent**, with the holder name as the title and the running command as the detail.
- While the agent holds control, the user cannot change the session: the adjustment panels,
  Undo, Redo, Versions changes, image switching, import, delete, rating, export, and project open
  and create are off. The user can still browse, zoom and pan, search, and read History and
  Versions.
- **Take control** (the cancel button of the agent task) gives control back to the user at once.
  The next agent command fails with `-32004`, and `control.acquire` fails with `-32004` too.
- **Return control** (on the canceled agent task) lets the agent call `control.acquire` again.
- `control.release` gives control back. The agent task ends as succeeded.
- Control does not belong to a connection. A second agent gets `-32005` while another holder
  has control.

## Tasks

Long operations return a task id at once. `tasks.wait` waits for the end of a task.

| Task id | Started by | Kind |
| --- | --- | --- |
| `import-N` | `library.import` | `import` |
| `export-N` | `export.run` | `export` |
| `bgtask-N` | `library.rate` with several photos, editor saves, and other background work | For example `ratingUpdate`, `editorSave`, `agentControl` |

The numbers start at 1 in each process. A task object has `task_id`, `kind`, `state`
(`queued`, `running`, `canceling`, `succeeded`, `failed`, or `canceled`), `progress` (percent,
-1 when unknown), and `message`. Import and export tasks also have `counts`.

- `tasks.wait {"task_id": "export-1"}` waits for one task.
- `tasks.wait {"idle": true}` waits until no import, export, or background task runs. Use it
  before you read results that a task writes, and before `session stop`.
- `tasks.list` lists the tasks that have not finished.

## Notifications

The host sends notifications to every connected client. `alcedo-cli watch` prints them.

| Notification | Params | When |
| --- | --- | --- |
| `task.updated` | A task object | A task changes and has not finished. |
| `task.finished` | A task object | A task finishes. |
| `session.state_changed` | `state`, `element_id`, `image_id` | The editor state or the open image changes. |
| `history.changed` | `element_id`, `image_id`, `head_commit`, `version_id` | The history head or the active Version changes, by the agent or by the user. |
| `control.changed` | `held`, `holder_name`, `revoked_by_user` | Agent control changes (GUI host only). |

Editor states: `NoImage`, `Acquiring`, `Loading`, `Interactive`, `Saving`, `Switching`,
`RetainedImageFailure`, `Failed`, `ShuttingDown`. Edits are possible in `Interactive`.

## Photos: element id and image id

`library.list` returns an `element_id` and an `image_id` for each photo. The commands take the
`element_id`. In a project made by import only, the two are equal; do not count on it.

## History and Versions

Each photo has one or more **Versions**. A Version is a line of history commits. The head of the
active Version is the current state of the photo.

- `editor.set` makes one commit when the value changes. `editor.batch_set` makes one commit for
  each changed field. The author of an agent commit is `agent`; the History panel shows the
  `note` of the call.
- `editor.undo` and `editor.redo` move the head along the commits of the active Version.
- `versions.create` starts a new Version at the image as imported. `versions.branch` starts a
  new Version at a commit of the active Version. `versions.checkout` switches the active
  Version.
- `editor.close`, `editor.open` of another photo, `project.save`, and `session.shutdown` save the
  open edits.

## UI values

All editor values are **UI values**: the values that the editor panels show, in the same
units. `editor.catalog` gives the range, step, and reset value of each field; see
[Editor fields](reference/editor_fields.md). A value outside the range is rejected with
`-32602`, not clamped.

A photo that has no edits starts with the default look of Alcedo Studio, not with the reset
values of the catalog: exposure +1.5, contrast 15, saturation 30. Read the current values with
`editor.get` before you change them. `editor.set` writes an absolute value; it does not add to
the current value.

Each field belongs to one node of the image: crop, white balance, and the RAW settings belong to
the Develop node, `odt` belongs to the display transform node, and the color fields belong to a
Color Grade node. The editor commands find the node themselves; a client does not select a node.
In the GUI session, a field that the selected node supports goes to that node, and all other
fields go to their own node. Thus, an agent can crop while the user has a Color Grade node
selected, and the selection does not change. A write of some properties of an object field
keeps the other properties at their current values.
