#!/usr/bin/env python3

import argparse
import importlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from clip_labeling import (
    CLIP_DEFAULT_CLASSES,
    DEFAULT_MAX_IN_FLIGHT,
    classify_images_by_prototypes,
    list_images,
    load_text_prototypes_cache,
    load_or_build_label_prototypes,
    text_prototypes_cache_path,
)
from semantic_client import (
    cargo_run_command,
    default_runtime_device,
    ensure_dependencies,
    start_server,
    stop_server,
    wait_until_ready,
)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Standalone album TUI: regenerate labels json from testdata and browse "
            "with arrow keys. Press Enter to open selected image."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="gRPC host")
    parser.add_argument("--port", default=50051, type=int, help="gRPC port")
    parser.add_argument(
        "--testdata-dir",
        default=str(repo_root / "testdata"),
        help="image directory to classify",
    )
    parser.add_argument(
        "--output-json",
        default=str(repo_root / "testdata" / "album_labels.json"),
        help="output json index path",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="do not spawn cargo run; use existing server",
    )
    parser.add_argument(
        "--device",
        default=default_runtime_device(),
        help="PUERH_MIND_DEVICE value used when spawning server",
    )
    parser.add_argument(
        "--top-k",
        default=3,
        type=int,
        help="store top-k scores for each image (>=1)",
    )
    parser.add_argument(
        "--request-size",
        default=DEFAULT_MAX_IN_FLIGHT,
        type=int,
        help="max in-flight image embed requests (>=1)",
    )
    return parser.parse_args()


def ensure_tui_dependencies():
    missing = []
    try:
        import rich  # noqa: F401
    except ImportError:
        missing.append("rich")

    try:
        import textual  # noqa: F401
    except ImportError:
        missing.append("textual")

    if importlib.util.find_spec("PIL") is None:
        missing.append("pillow")

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"missing required Python packages: {joined}\n"
            "install with: pip install textual pillow"
        )


def write_album_index(output_json_path, address, testdata_dir, entries):
    success_count = sum(1 for item in entries if item.get("error") is None)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "address": address,
        "testdata_dir": str(testdata_dir.resolve()),
        "classes": CLIP_DEFAULT_CLASSES,
        "total": len(entries),
        "success_count": success_count,
        "failure_count": len(entries) - success_count,
        "items": entries,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def refresh_album_index(
    host="127.0.0.1",
    port=50051,
    testdata_dir=None,
    no_spawn=False,
    device=None,
    top_k=3,
    request_size=DEFAULT_MAX_IN_FLIGHT,
    output_json_path=None,
    progress_callback=None,
):
    if top_k < 1:
        raise RuntimeError("--top-k must be >= 1")
    if request_size < 1:
        raise RuntimeError("--request-size must be >= 1")
    if device is None:
        device = default_runtime_device()

    address = f"{host}:{port}"
    repo_root = Path(__file__).resolve().parents[1]
    testdata_dir = Path(testdata_dir or (repo_root / "testdata"))
    output_json_path = Path(output_json_path or (testdata_dir / "album_labels.json"))

    if not testdata_dir.exists():
        raise RuntimeError(f"testdata dir not found: {testdata_dir}")

    ensure_dependencies(repo_root, require_cargo=False, require_grpc=True)

    server_proc = None
    owns_server = False

    try:
        if no_spawn:
            if not wait_until_ready(address, timeout_sec=3):
                raise RuntimeError(f"server is not reachable at {address}")
            print(f"using existing server at {address}")
        else:
            if wait_until_ready(address, timeout_sec=1):
                print(f"using existing server at {address} (skip spawn)")
            else:
                ensure_dependencies(repo_root, require_cargo=True, require_grpc=True)
                print(
                    f"starting rust server with {' '.join(cargo_run_command(repo_root))} "
                    f"and PUERH_MIND_DEVICE={device} at {repo_root} ..."
                )
                server_proc = start_server(
                    repo_root,
                    address,
                    device=device,
                    quiet_stdio=True,
                    rust_log="off",
                )
                owns_server = True
                print(f"server started at {address}")

        images = list_images(testdata_dir)
        total_images = len(images)
        if progress_callback:
            progress_callback(0, total_images, "waiting for image embeddings")

        label_prototypes, text_cache_path, reused_cache = load_or_build_label_prototypes(
            address,
            CLIP_DEFAULT_CLASSES,
            testdata_dir,
        )
        if reused_cache:
            print(f"loaded text prototypes from cache: {text_cache_path}")
        else:
            print(f"generated text prototypes and cached to: {text_cache_path}")
        entries = []

        def on_image_result(completed, total_count, image_path, _result):
            if progress_callback:
                progress_callback(
                    completed,
                    total_count,
                    f"embedding images {completed}/{total_count}: {image_path.name}",
                )

        results = classify_images_by_prototypes(
            address,
            images,
            label_prototypes,
            max_in_flight=request_size,
            result_callback=on_image_result,
        )

        for idx, result in enumerate(results, start=1):
            image_path = result["image_path"]
            error = result["error"]
            try:
                if error:
                    raise RuntimeError(error)

                scores = result["scores"]
                best_label, best_score = scores[0]
                top_scores = scores[:top_k]
                entries.append(
                    {
                        "index": idx,
                        "name": image_path.name,
                        "path": str(image_path.resolve()),
                        "label": best_label,
                        "score": best_score,
                        "top_scores": [
                            {"label": label, "score": score} for label, score in top_scores
                        ],
                        "error": None,
                    }
                )
            except Exception as exc:
                entries.append(
                    {
                        "index": idx,
                        "name": image_path.name,
                        "path": str(image_path.resolve()),
                        "label": None,
                        "score": None,
                        "top_scores": [],
                        "error": str(exc),
                    }
                )

        write_album_index(output_json_path, address, testdata_dir, entries)
        return output_json_path

    finally:
        if owns_server:
            print("\nstopping rust server ...")
            stop_server(server_proc)


def load_album_index(output_json_path):
    if not output_json_path.exists():
        raise RuntimeError(f"album index json not found: {output_json_path}")
    payload = json.loads(output_json_path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise RuntimeError("album index json has no items")
    return payload


def can_reuse_album_index(output_json_path, testdata_dir):
    output_json_path = Path(output_json_path)
    testdata_dir = Path(testdata_dir)

    if not output_json_path.exists():
        return False

    try:
        payload = json.loads(output_json_path.read_text(encoding="utf-8"))
        items = payload.get("items")
        if not isinstance(items, list):
            return False

        if payload.get("classes") != CLIP_DEFAULT_CLASSES:
            return False

        cache_path = text_prototypes_cache_path(testdata_dir)
        if load_text_prototypes_cache(cache_path, CLIP_DEFAULT_CLASSES) is None:
            return False

        image_count = len(list_images(testdata_dir))
        return len(items) == image_count
    except Exception:
        return False


def open_image(path_text):
    path = Path(path_text)
    if not path.exists():
        raise RuntimeError(f"image not found: {path}")

    if os.name == "nt":
        os.startfile(str(path))
        return

    raise RuntimeError("only Windows is supported for open action in this TUI")


def format_item_label(item):
    return item.get("label") or ("ERROR" if item.get("error") else "-")


def format_item_score(item):
    score = item.get("score")
    if isinstance(score, (int, float)):
        return f"{score:.6f}"
    return "-"


def item_filter_tag(item):
    if item.get("error"):
        return "ERROR"
    return item.get("label") or "UNLABELED"


def build_tag_counts(items):
    counts = {}
    for item in items:
        tag = item_filter_tag(item)
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def render_color_preview(path_text, max_width=72, max_height=64):
    from rich.text import Text

    image_module = importlib.import_module("PIL.Image")

    path = Path(path_text)
    if not path.exists():
        return Text("(image not found)")

    with image_module.open(path) as image:
        image = image.convert("RGB")
        src_w, src_h = image.size
        if src_w <= 0 or src_h <= 0:
            return Text("(invalid image)")

        target_w = max(16, min(max_width, src_w))
        target_h = int((src_h / src_w) * target_w)
        target_h = max(10, target_h)

        if target_h > max_height:
            ratio = max_height / float(target_h)
            target_h = max_height
            target_w = max(16, int(target_w * ratio))

        # Render with half-block glyphs: each terminal row represents two image rows.
        # This roughly doubles vertical detail compared with plain ASCII shading.
        if target_h % 2 != 0:
            target_h += 1

        image = image.resize((target_w, target_h))
        pixels = list(image.getdata())

    renderable = Text()
    for row in range(0, target_h, 2):
        top_start = row * target_w
        bottom_start = (row + 1) * target_w
        top_pixels = pixels[top_start : top_start + target_w]
        bottom_pixels = pixels[bottom_start : bottom_start + target_w]

        for top, bottom in zip(top_pixels, bottom_pixels):
            top_r, top_g, top_b = top
            bot_r, bot_g, bot_b = bottom
            style = (
                f"#{top_r:02x}{top_g:02x}{top_b:02x} "
                f"on #{bot_r:02x}{bot_g:02x}{bot_b:02x}"
            )
            renderable.append("▀", style=style)
        renderable.append("\n")

    return renderable


def build_progress_callback(progress, task_id):
    def on_progress(step, total_steps, message):
        progress.update(
            task_id,
            total=total_steps,
            completed=step,
            description=f"[cyan]{message}",
        )

    return on_progress


def refresh_with_progress(**kwargs):
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("[cyan]waiting for image embeddings", total=1)
        callback = build_progress_callback(progress, task_id)
        return refresh_album_index(progress_callback=callback, **kwargs)


def run_tui(payload):
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Button, DataTable, Footer, Header, Static

    class AlbumApp(App):
        CSS = """
        Screen {
            background: #0f172a;
            color: #e2e8f0;
        }

        Header {
            background: #1d4ed8;
            color: #f8fafc;
            text-style: bold;
        }

        Footer {
            background: #0b1120;
            color: #cbd5e1;
        }

        #main {
            height: 1fr;
            padding: 1;
            background: #0b1220;
        }

        #album-table {
            width: 3fr;
            border: round #38bdf8;
            background: #111827;
            height: 1fr;
        }

        #right-pane {
            width: 2fr;
            min-width: 58;
            padding-left: 1;
            height: 1fr;
        }

        #filters-title {
            text-style: bold;
            color: #a7f3d0;
            padding: 0 1;
        }

        #tag-scroll {
            height: 8;
            border: round #34d399;
            background: #111827;
            padding: 0 1;
        }

        .filter-btn {
            width: 1fr;
            margin: 0 0 1 0;
        }

        .active-filter {
            background: #f59e0b;
            color: #0f172a;
            text-style: bold;
        }

        #detail {
            height: 9;
            border: round #f59e0b;
            background: #111827;
            padding: 1 2;
            margin-top: 1;
            overflow: auto;
        }

        #preview {
            height: 1fr;
            border: round #a78bfa;
            background: #111827;
            padding: 1;
            margin-top: 1;
            overflow: auto;
        }
        """

        BINDINGS = [
            Binding("enter", "open_selected", "Open", show=True),
            Binding("+", "zoom_in", "Zoom+", show=True),
            Binding("-", "zoom_out", "Zoom-", show=True),
            Binding("q", "quit", "Quit", show=True),
        ]

        def __init__(self, album_payload):
            super().__init__()
            self.album_payload = album_payload
            self.items = list(album_payload["items"])
            self.visible_items = list(self.items)
            self.current_index = 0
            self.current_filter = "ALL"
            self.preview_cache = {}
            self.preview_scale = 1.0

            self.tag_counts = build_tag_counts(self.items)
            self.filter_tags = ["ALL"] + sorted(self.tag_counts.keys())
            self.button_to_tag = {}
            self.tag_to_button = {}

            for idx, tag in enumerate(self.filter_tags):
                button_id = f"filter-{idx}"
                self.button_to_tag[button_id] = tag
                self.tag_to_button[tag] = button_id

        def format_filter_label(self, tag):
            if tag == "ALL":
                return f"ALL ({len(self.items)})"
            return f"{tag} ({self.tag_counts.get(tag, 0)})"

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                yield DataTable(id="album-table")
                with Vertical(id="right-pane"):
                    yield Static("Tag Filters", id="filters-title")
                    with VerticalScroll(id="tag-scroll"):
                        for tag in self.filter_tags:
                            yield Button(
                                self.format_filter_label(tag),
                                id=self.tag_to_button[tag],
                                classes="filter-btn",
                            )
                    yield Static("", id="detail")
                    yield Static("", id="preview")
            yield Footer()

        def on_mount(self):
            table = self.query_one("#album-table", DataTable)
            table.add_columns("#", "Name", "Label", "Score")
            table.cursor_type = "row"
            table.zebra_stripes = True

            self.refresh_table()
            self.update_filter_button_style()

        def refresh_table(self):
            table = self.query_one("#album-table", DataTable)
            table.clear(columns=False)

            for idx, item in enumerate(self.visible_items, start=1):
                table.add_row(
                    str(item.get("index", idx)),
                    item.get("name", "-"),
                    format_item_label(item),
                    format_item_score(item),
                )

            if self.visible_items:
                self.current_index = 0
                table.move_cursor(row=0, column=0)
                self.update_right_panel(0)
            else:
                self.current_index = -1
                self.query_one("#detail", Static).update(
                    f"[b]filter[/b]: {self.current_filter}\n\nNo items match this filter."
                )
                self.query_one("#preview", Static).update("No image")

        def update_filter_button_style(self):
            for button_id, tag in self.button_to_tag.items():
                button = self.query_one(f"#{button_id}", Button)
                button.remove_class("active-filter")
                if tag == self.current_filter:
                    button.add_class("active-filter")

        def apply_filter(self, tag):
            self.current_filter = tag
            if tag == "ALL":
                self.visible_items = list(self.items)
            else:
                self.visible_items = [
                    item for item in self.items if item_filter_tag(item) == tag
                ]

            self.refresh_table()
            self.update_filter_button_style()
            self.notify(f"filter: {tag} ({len(self.visible_items)})", title="Album")

        def on_button_pressed(self, event):
            button_id = event.button.id
            if button_id in self.button_to_tag:
                self.apply_filter(self.button_to_tag[button_id])

        def on_data_table_row_highlighted(self, event):
            if event.cursor_row is None:
                return
            if event.cursor_row < 0 or event.cursor_row >= len(self.visible_items):
                return

            self.current_index = event.cursor_row
            self.update_right_panel(self.current_index)

        def update_right_panel(self, index):
            if not self.visible_items:
                self.query_one("#detail", Static).update("No items")
                self.query_one("#preview", Static).update("No image")
                return

            item = self.visible_items[index]
            lines = [
                f"[b]{item.get('name', '-')}[/b]",
                "",
                f"filter: {self.current_filter}",
                f"label: {item.get('label', '-')}",
                f"score: {format_item_score(item)}",
                f"index: {index + 1}/{len(self.visible_items)}",
                "",
            ]

            error = item.get("error")
            if error:
                lines.append("error:")
                lines.append(error)
            else:
                lines.append("top scores:")
                for top in item.get("top_scores", [])[:8]:
                    label = top.get("label", "-")
                    score = top.get("score")
                    if isinstance(score, (int, float)):
                        lines.append(f"- {label}: {score:.6f}")
                    else:
                        lines.append(f"- {label}: -")

            lines.extend(
                [
                    "",
                    "path:",
                    item.get("path", "-"),
                    "",
                    "json:",
                    str(self.album_payload.get("_json_path", "-")),
                ]
            )

            self.query_one("#detail", Static).update("\n".join(lines))

            image_path = item.get("path", "")
            preview_widget = self.query_one("#preview", Static)
            width = max(16, preview_widget.size.width - 2)
            rows = max(8, preview_widget.size.height - 2)

            scaled_width = min(180, int(width * self.preview_scale))
            scaled_rows = min(90, int(rows * self.preview_scale))

            preview_key = (image_path, scaled_width, scaled_rows)
            if preview_key not in self.preview_cache:
                try:
                    self.preview_cache[preview_key] = render_color_preview(
                        image_path,
                        max_width=scaled_width,
                        max_height=scaled_rows * 2,
                    )
                except Exception as exc:
                    self.preview_cache[preview_key] = f"preview unavailable: {exc}"

            preview_widget.update(self.preview_cache[preview_key])

        def action_zoom_in(self):
            self.preview_scale = min(2.5, self.preview_scale + 0.25)
            if self.visible_items and self.current_index >= 0:
                self.update_right_panel(self.current_index)
            self.notify(f"preview zoom: {self.preview_scale:.2f}x", title="Album")

        def action_zoom_out(self):
            self.preview_scale = max(0.75, self.preview_scale - 0.25)
            if self.visible_items and self.current_index >= 0:
                self.update_right_panel(self.current_index)
            self.notify(f"preview zoom: {self.preview_scale:.2f}x", title="Album")

        def action_open_selected(self):
            if not self.visible_items:
                return

            if self.current_index < 0 or self.current_index >= len(self.visible_items):
                return

            item = self.visible_items[self.current_index]
            try:
                open_image(item.get("path", ""))
                self.notify(f"opened: {item.get('name', '-')}", title="Album")
            except Exception as exc:
                self.notify(str(exc), title="Open failed", severity="error")

    AlbumApp(payload).run()


def main():
    args = parse_args()

    ensure_tui_dependencies()

    testdata_dir = Path(args.testdata_dir)
    output_json_path = Path(args.output_json)

    if can_reuse_album_index(output_json_path, testdata_dir):
        print(f"reusing existing album index: {output_json_path}")
    else:
        output_json_path = refresh_with_progress(
            host=args.host,
            port=args.port,
            testdata_dir=str(testdata_dir),
            no_spawn=args.no_spawn,
            device=args.device,
            top_k=args.top_k,
            request_size=args.request_size,
            output_json_path=str(output_json_path),
        )

    payload = load_album_index(Path(output_json_path))
    payload["_json_path"] = str(Path(output_json_path).resolve())
    run_tui(payload)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
