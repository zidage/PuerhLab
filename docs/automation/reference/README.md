# Command reference

Protocol version 1, Alcedo Studio 0.3.2. `alcedo-cli schema` prints the same list from a running
session.

Each section has the description, the parameters, and the result. These parts are generated
from [commands.json](commands.json); do not edit them by hand. The errors and the example are
written by hand. The examples ran with the release build. Every command also accepts
`timeout_ms`; see [cli.md](../cli.md#call).

"Changes state: yes" means that a GUI session needs agent control for the command
([Concepts](../concepts.md#agent-control-gui-host)).

| Group | Commands |
| --- | --- |
| [Session](session.md) | `session.ping`, `session.describe`, `state.get`, `session.shutdown` |
| [Control](control.md) | `control.acquire`, `control.release`, `control.get` |
| [Project](project.md) | `project.create`, `project.open`, `project.save`, `project.close` |
| [Library](library.md) | `library.import`, `library.folders`, `library.list`, `library.thumbnail`, `library.rate`, `library.delete`, `library.selection.get`, `library.selection.set` |
| [Tasks](tasks.md) | `tasks.list`, `tasks.wait` |
| [Editor](editor.md) | `editor.open`, `editor.close`, `editor.catalog`, `editor.get`, `editor.set`, `editor.batch_set`, `editor.undo`, `editor.redo`, `editor.history`, `editor.actions` |
| [Render](render.md) | `render.preview` |
| [Versions](versions.md) | `versions.list`, `versions.create`, `versions.branch`, `versions.checkout`, `versions.rename`, `versions.remove` |
| [Export](export.md) | `export.options`, `export.run`, `export.status` |
| [Search](search.md) | `search.query` |

[Editor fields](editor_fields.md) lists every field of `editor.set`.
