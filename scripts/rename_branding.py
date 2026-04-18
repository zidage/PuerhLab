#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SKIP_DIRS = {".git"}
TEXT_SCAN_LIMIT_BYTES = 5 * 1024 * 1024
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")

ALNUM_BOUNDARY = r"[A-Za-z0-9]"
EXTENSION_PATTERN = re.compile(r"(?i)\.alcd(?![A-Za-z0-9])")
DISPLAY_LAB_PATTERN = re.compile(
    rf"(?i)(?<!{ALNUM_BOUNDARY})(?:alcedo\s+lab|pu[\s-]+erh\s+lab)(?!{ALNUM_BOUNDARY})"
)
LAB_TOKEN_PATTERN = re.compile(
    rf"(?i)(?<!{ALNUM_BOUNDARY})(?:alcedo|alcedo[_-]?lab|pu[\s_-]?erh[_-]?lab)(?!{ALNUM_BOUNDARY})"
)
ROOT_TOKEN_PATTERN = re.compile(
    rf"(?i)(?<!{ALNUM_BOUNDARY})(?:alcedo|pu[\s_-]?erh)(?!{ALNUM_BOUNDARY})"
)


@dataclass(frozen=True)
class DecodedText:
    text: str
    encoding: str


@dataclass(frozen=True)
class ContentChange:
    path: Path
    match_count: int


@dataclass(frozen=True)
class RenameChange:
    path: Path
    new_name: str
    match_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a project for legacy alcedo-based names, print statistics, and optionally "
            "rewrite file contents plus file/directory names."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=str(REPO_ROOT),
        help="Project root to scan. Defaults to the repository root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the scan summary; do not rewrite any files or paths.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Extra directory name or relative path to skip. "
            "`.git` is always skipped and cannot be re-enabled."
        ),
    )
    parser.add_argument(
        "--summary-limit",
        type=int,
        default=200,
        help="Maximum number of content/path entries to print in each summary section.",
    )
    parser.add_argument(
        "--verbose-skips",
        action="store_true",
        help="Print files skipped because they look binary or exceed the text scan size limit.",
    )
    return parser.parse_args()


def preserve_word_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original[:1].isupper() and original[1:].islower():
        return replacement.capitalize()
    if any(ch.isupper() for ch in original):
        return replacement.capitalize()
    return replacement.lower()


def preserve_phrase_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    return replacement


def replace_extension(text: str) -> tuple[str, int]:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.isupper():
            return ".ALCD"
        if token.islower():
            return ".alcd"
        return ".alcd"

    return EXTENSION_PATTERN.subn(repl, text)


def replace_display_lab(text: str) -> tuple[str, int]:
    def repl(match: re.Match[str]) -> str:
        return preserve_phrase_case(match.group(0), "Alcedo Studio")

    return DISPLAY_LAB_PATTERN.subn(repl, text)


def replace_lab_token(text: str, replacement: str) -> tuple[str, int]:
    def repl(match: re.Match[str]) -> str:
        return preserve_word_case(match.group(0), replacement)

    return LAB_TOKEN_PATTERN.subn(repl, text)


def replace_root_token(text: str, replacement: str) -> tuple[str, int]:
    def repl(match: re.Match[str]) -> str:
        return preserve_word_case(match.group(0), replacement)

    return ROOT_TOKEN_PATTERN.subn(repl, text)


def rewrite_content(text: str) -> tuple[str, int]:
    updated = text
    total = 0
    for transform in (
        replace_extension,
        replace_display_lab,
        lambda value: replace_lab_token(value, "alcedo"),
        lambda value: replace_root_token(value, "alcedo"),
    ):
        updated, count = transform(updated)
        total += count
    return updated, total


def rewrite_path_name(name: str) -> tuple[str, int]:
    updated = name
    total = 0
    for transform in (
        replace_extension,
        lambda value: replace_lab_token(value, "alcedo"),
        lambda value: replace_root_token(value, "alcedo"),
    ):
        updated, count = transform(updated)
        total += count
    return updated, total


def should_skip_dir(relative_dir: Path, dir_name: str, extra_skips: set[str]) -> bool:
    if dir_name in DEFAULT_SKIP_DIRS:
        return True

    relative_posix = relative_dir.as_posix()
    return dir_name in extra_skips or relative_posix in extra_skips


def decode_text_file(path: Path) -> DecodedText | None:
    raw = path.read_bytes()
    if len(raw) > TEXT_SCAN_LIMIT_BYTES:
        return None
    if b"\x00" in raw:
        return None
    for encoding in TEXT_ENCODINGS:
        try:
            return DecodedText(raw.decode(encoding), encoding)
        except UnicodeDecodeError:
            continue
    return None


def write_text_file(path: Path, decoded: DecodedText, text: str) -> None:
    path.write_bytes(text.encode(decoded.encoding))


def iter_project_paths(root: Path, extra_skips: set[str]) -> Iterable[Path]:
    for current_root, dirnames, filenames in root.walk(top_down=True):
        relative_root = current_root.relative_to(root)
        kept_dirs: list[str] = []
        for dir_name in dirnames:
            relative_dir = (relative_root / dir_name) if relative_root != Path(".") else Path(dir_name)
            if should_skip_dir(relative_dir, dir_name, extra_skips):
                continue
            kept_dirs.append(dir_name)
        dirnames[:] = kept_dirs

        for dir_name in dirnames:
            yield current_root / dir_name
        for file_name in filenames:
            yield current_root / file_name


def collect_changes(
    root: Path,
    extra_skips: set[str],
    verbose_skips: bool,
) -> tuple[list[ContentChange], list[RenameChange], list[Path]]:
    content_changes: list[ContentChange] = []
    rename_changes: list[RenameChange] = []
    skipped_files: list[Path] = []

    for path in iter_project_paths(root, extra_skips):
        new_name, rename_count = rewrite_path_name(path.name)
        if rename_count > 0 and new_name != path.name:
            rename_changes.append(RenameChange(path=path, new_name=new_name, match_count=rename_count))

        if not path.is_file():
            continue

        decoded = decode_text_file(path)
        if decoded is None:
            if verbose_skips:
                skipped_files.append(path)
            continue

        updated_text, content_count = rewrite_content(decoded.text)
        if content_count > 0 and updated_text != decoded.text:
            content_changes.append(ContentChange(path=path, match_count=content_count))

    return content_changes, rename_changes, skipped_files


def print_summary(
    root: Path,
    content_changes: list[ContentChange],
    rename_changes: list[RenameChange],
    skipped_files: list[Path],
    summary_limit: int,
) -> None:
    total_content_matches = sum(item.match_count for item in content_changes)
    total_path_matches = sum(item.match_count for item in rename_changes)
    total_matches = total_content_matches + total_path_matches

    print(f"扫描根目录: {root}")
    print(
        "扫描完成: "
        f"{total_matches} 个匹配项, "
        f"其中内容匹配 {total_content_matches} 个, "
        f"路径匹配 {total_path_matches} 个。"
    )
    print(f"内容命中文件数: {len(content_changes)}")
    print(f"待重命名路径数: {len(rename_changes)}")

    if content_changes:
        print("\n内容命中清单:")
        for item in sorted(content_changes, key=lambda entry: entry.path.as_posix())[:summary_limit]:
            print(f"  [content] {item.path.relative_to(root)} ({item.match_count})")
        if len(content_changes) > summary_limit:
            print(f"  ... 还有 {len(content_changes) - summary_limit} 个内容命中文件未展开")

    if rename_changes:
        print("\n路径重命名清单:")
        for item in sorted(rename_changes, key=lambda entry: entry.path.as_posix())[:summary_limit]:
            print(f"  [rename]  {item.path.relative_to(root)} -> {item.new_name} ({item.match_count})")
        if len(rename_changes) > summary_limit:
            print(f"  ... 还有 {len(rename_changes) - summary_limit} 个待重命名路径未展开")

    if skipped_files:
        print(f"\n跳过的疑似二进制或超大文件: {len(skipped_files)}")
        for path in skipped_files[:summary_limit]:
            print(f"  [skip]    {path.relative_to(root)}")
        if len(skipped_files) > summary_limit:
            print(f"  ... 还有 {len(skipped_files) - summary_limit} 个跳过文件未展开")


def apply_content_changes(root: Path, content_changes: list[ContentChange]) -> int:
    updated_files = 0
    for item in content_changes:
        decoded = decode_text_file(item.path)
        if decoded is None:
            raise RuntimeError(f"无法重新读取文本文件: {item.path}")
        updated_text, _ = rewrite_content(decoded.text)
        if updated_text != decoded.text:
            write_text_file(item.path, decoded, updated_text)
            updated_files += 1
    print(f"\n已更新文件内容: {updated_files}")
    return updated_files


def apply_rename_changes(root: Path, rename_changes: list[RenameChange]) -> int:
    renamed_paths = 0
    ordered_changes = sorted(
        rename_changes,
        key=lambda item: (len(item.path.relative_to(root).parts), item.path.as_posix()),
        reverse=True,
    )
    for item in ordered_changes:
        if not item.path.exists():
            continue
        destination = item.path.with_name(item.new_name)
        if destination.exists():
            raise FileExistsError(f"目标路径已存在, 无法重命名: {destination}")
        item.path.rename(destination)
        renamed_paths += 1
    print(f"已重命名路径: {renamed_paths}")
    return renamed_paths


def validate_root(root: Path) -> Path:
    resolved = root.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"扫描根目录不存在: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"扫描根路径不是目录: {resolved}")
    return resolved


def main() -> int:
    args = parse_args()

    try:
        root = validate_root(Path(args.root))
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2

    extra_skips = {item.replace("\\", "/").strip("/.") for item in args.exclude_dir if item.strip()}
    try:
        content_changes, rename_changes, skipped_files = collect_changes(
            root=root,
            extra_skips=extra_skips,
            verbose_skips=args.verbose_skips,
        )
        print_summary(
            root=root,
            content_changes=content_changes,
            rename_changes=rename_changes,
            skipped_files=skipped_files,
            summary_limit=max(args.summary_limit, 1),
        )

        if args.dry_run:
            print("\nDry run 模式: 未执行任何写入或重命名。")
            return 0

        apply_content_changes(root, content_changes)
        apply_rename_changes(root, rename_changes)
        print("替换与重命名已完成。")
        return 0
    except Exception as exc:
        print(f"\n执行失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
