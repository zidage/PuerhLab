# alcedo-cli and the automation protocol

`alcedo-cli` controls Alcedo Studio from a terminal, a script, or an AI agent. It talks to a
running `alcedo_main` process through a local socket. Each request is one JSON-RPC 2.0 command,
for example `editor.set` or `export.run`.

You can use it in two ways:

- **Headless:** `alcedo-cli session start --headless` starts `alcedo_main` without a window.
  Use it for batch work and scripts.
- **GUI:** the normal Alcedo Studio window also serves a session named `gui`. An agent can edit
  in the window while you watch. The agent takes control first, and you can take control back at
  any time.

## Pages

| Page | Contents |
| --- | --- |
| [Installation](installation.md) | Where `alcedo-cli` and `alcedo_main` are, and how to check the setup. |
| [Concepts](concepts.md) | Sessions, session files, hosts, agent control, tasks, notifications, history and Versions, UI values. |
| [Command-line interface](cli.md) | Global options, `call`, `schema`, `session start/stop/list/prune`, `watch`, output, exit codes, error codes. |
| [Workflows](workflows.md) | Complete examples: import and rate, edit and preview, Versions, export, search, edit in the GUI. |
| [Known limits](limits.md) | Behavior to know about, and the open issues. |
| [Command reference](reference/README.md) | Every command with its parameters, result, errors, and an example. |
| [Editor fields](reference/editor_fields.md) | Every field of `editor.set` with its range and reset value. |

## Quick start

```bash
alcedo-cli session start --headless --create D:/Photos,trip
alcedo-cli call library.import '{"folder":"D:/Photos/2026-trip"}'
alcedo-cli call tasks.wait '{"idle":true}'
alcedo-cli call library.list '{"limit":5}'
alcedo-cli call editor.open '{"element_id":1}'
alcedo-cli call editor.set '{"field":"exposure","value":1.8}'
alcedo-cli call render.preview '{"out_dir":"D:/Photos/previews"}'
alcedo-cli call export.run '{"element_ids":[1],"out_dir":"D:/Photos/export"}'
alcedo-cli call tasks.wait '{"idle":true}'
alcedo-cli session stop
```

`session start --create D:/Photos,trip` creates the project `D:/Photos/trip.alcd`. With only one
live session, every command goes to it. See [Workflows](workflows.md) for the details of each
step.

## Agent skill

The skill `alcedo` in the [alcedo-skills](https://github.com/zidage/alcedo-skills) repository
gives an agent harness (for example Claude Code) these pages and the rules to use the CLI
well. It also has an optional script that measures a preview (brightness, color, and tilt).

## Keep this reference up to date

The parameter and result tables of the command reference and the editor field tables are
generated from `reference/commands.json` and `reference/editor_catalog.json`:

- `AutomationReferenceDocsTest` fails when these files differ from the schemas of the build. The
  failure message names the new files.
- `scripts/automation/generate_cli_reference.py --schema <new commands.json> --catalog <new
  editor_catalog.json>` copies them and rewrites the generated blocks.
- `generate_cli_reference.py --check` (also the test `AutomationReferenceDocsGenerated`) fails
  when a block is not up to date or a command has no section. Write the errors and the example
  of a new command by hand.
