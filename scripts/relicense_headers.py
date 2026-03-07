#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = (
    REPO_ROOT / "pu-erh_lab" / "src",
    REPO_ROOT / "pu-erh_lab" / "tests",
)
CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
}
PURE_THIRD_PARTY_PATHS = {
    Path("pu-erh_lab/src/config/CLFs/hilite_recon.cc"),
    Path("pu-erh_lab/src/include/edit/operators/basic/camera_matrices.hpp"),
    Path("pu-erh_lab/tests/leak_detector/memory_leak_detector.hpp"),
}
PURE_THIRD_PARTY_MARKERS = (
    "creativecommons.org/licenses/by-sa/4.0",
    "Copied from https://stackoverflow.com/questions/29174938/googletest-and-memory-leaks",
    "Copyright (c) 2025 Muperman",
    "hilite_recon.cc is free software",
    "Extract from Adobe DNG Converter using the tool from https://github.com/ilia3101/ExtractAdobeCameraMatrices",
)
MIXED_PROVENANCE_MARKERS = (
    "darktable developers",
    "Adapted from https://github.com/LuisSR/RCD-Demosaicing",
    "RATIO CORRECTED DEMOSAICING",
    "Luis Sanz Rodr",
    "This file contains GPLv3-derived",
    "by Jed Smith: https://github.com/jedypod/open-display-transform",
)


@dataclass(frozen=True)
class FileResult:
    status: str
    path: Path
    note: str = ""


@dataclass(frozen=True)
class ParsedLine:
    text: str
    ending: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate pu-erh_lab source/test headers to GPL-3.0-only."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Report files that would change.")
    group.add_argument("--write", action="store_true", help="Rewrite file headers in place.")
    return parser.parse_args()


def iter_target_files() -> list[Path]:
    files: list[Path] = []
    for root in TARGET_DIRS:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in CODE_EXTENSIONS:
                files.append(path)
    return sorted(files)


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="")


def parse_lines(text: str) -> list[ParsedLine]:
    lines: list[ParsedLine] = []
    offset = 0
    for full_line in text.splitlines(keepends=True):
        if full_line.endswith("\r\n"):
            line_text = full_line[:-2]
            ending = "\r\n"
        elif full_line.endswith("\n") or full_line.endswith("\r"):
            line_text = full_line[:-1]
            ending = full_line[-1]
        else:
            line_text = full_line
            ending = ""
        lines.append(ParsedLine(line_text, ending, offset, offset + len(full_line)))
        offset += len(full_line)
    if not lines and text:
        lines.append(ParsedLine(text, "", 0, len(text)))
    return lines


def detect_header_newline(lines: list[ParsedLine]) -> str:
    for line in lines:
        if line.ending:
            return line.ending
    return "\n"


def extract_copyright_text(line_text: str) -> str | None:
    if not line_text.startswith("//"):
        return None
    payload = line_text[2:].strip()
    if not payload.startswith("Copyright") or "Yurun Zi" not in payload:
        return None
    return payload


def is_pure_third_party(rel_path: Path, text: str) -> bool:
    if rel_path in PURE_THIRD_PARTY_PATHS:
        return True
    return any(marker in text for marker in PURE_THIRD_PARTY_MARKERS)


def is_mixed_provenance(text: str) -> bool:
    preview = "\n".join(text.splitlines()[:80])
    return any(marker in preview for marker in MIXED_PROVENANCE_MARKERS)


def looks_like_legacy_mixed(lines: list[ParsedLine], start: int) -> bool:
    return (
        start + 2 < len(lines)
        and extract_copyright_text(lines[start].text) is not None
        and lines[start + 1].text.strip() == "//"
        and lines[start + 2].text.startswith("//  This file contains GPLv3-derived")
    )


def strip_existing_header(text: str) -> tuple[str | None, int]:
    lines = parse_lines(text)
    if not lines:
        return None, 0

    copyright_text = extract_copyright_text(lines[0].text)
    if copyright_text is None:
        return None, 0

    if looks_like_legacy_mixed(lines, 0):
        return copyright_text, lines[2].start

    if len(lines) > 1 and "SPDX-License-Identifier: GPL-3.0-only" in lines[1].text:
        idx = 2
        if idx < len(lines) and lines[idx].text == "//":
            idx += 1
            if (
                idx < len(lines)
                and lines[idx].text
                == "//  This file also contains material subject to the upstream notices below."
            ):
                idx += 1
        elif (
            idx < len(lines)
            and lines[idx].text
            == "//  Additional permission under GPLv3 section 7 applies; see the LICENSE file."
        ):
            idx += 1
        if idx < len(lines) and lines[idx].text == "":
            idx += 1
        if looks_like_legacy_mixed(lines, idx):
            idx += 2
        return copyright_text, lines[idx].start if idx < len(lines) else len(text)

    if any("Licensed under the Apache License, Version 2.0" in line.text for line in lines[:6]):
        idx = 1
        while idx < len(lines):
            if "limitations under the License." in lines[idx].text:
                idx += 1
                if idx < len(lines) and lines[idx].text == "":
                    idx += 1
                return copyright_text, lines[idx].start if idx < len(lines) else len(text)
            idx += 1

    return copyright_text, 0


def starts_with_unowned_comment(text: str) -> bool:
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#pragma once") or stripped.startswith("#include"):
            return False
        if stripped.startswith("/*"):
            return True
        if stripped.startswith("//"):
            return False
        return False
    return False


@lru_cache(maxsize=None)
def copyright_years(rel_path: str) -> str | None:
    result = subprocess.run(
        [
            "git",
            "log",
            "--follow",
            "--format=%ad",
            "--date=format:%Y",
            "--",
            rel_path,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    years = sorted({line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()})
    if not years:
        return None
    if len(years) == 1:
        return years[0]
    return f"{years[0]}-{years[-1]}"


def default_copyright_text(rel_path: Path) -> str | None:
    years = copyright_years(rel_path.as_posix())
    if years is None:
        return None
    return f"Copyright {years} Yurun Zi"


def build_header(copyright_text: str, mixed_provenance: bool, newline: str, has_body: bool) -> str:
    lines = [
        f"//  {copyright_text}",
        "//  SPDX-License-Identifier: GPL-3.0-only",
    ]
    if mixed_provenance:
        lines.extend(
            [
                "//",
                "//  This file also contains material subject to the upstream notices below.",
            ]
        )
    else:
        lines.append("//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.")

    header = newline.join(lines) + newline
    if has_body:
        header += newline
    return header


def rewrite_text(rel_path: Path, text: str) -> tuple[FileResult, str | None]:
    if is_pure_third_party(rel_path, text):
        return FileResult("skipped", rel_path, "third-party"), None

    bom = "\ufeff" if text.startswith("\ufeff") else ""
    body_text = text[len(bom) :]

    existing_copyright, body_start = strip_existing_header(body_text)
    remaining = body_text[body_start:]
    mixed = is_mixed_provenance(body_text)

    if existing_copyright is None and starts_with_unowned_comment(remaining):
        return FileResult("needs-manual-review", rel_path, "unrecognized leading comment block"), None

    copyright_text = existing_copyright or default_copyright_text(rel_path)
    if copyright_text is None:
        return FileResult("needs-manual-review", rel_path, "unable to determine copyright years"), None

    newline = detect_header_newline(parse_lines(body_text))
    new_text = bom + build_header(copyright_text, mixed, newline, bool(remaining)) + remaining

    if new_text == text:
        return FileResult("unchanged", rel_path), None
    return FileResult("updated", rel_path), new_text


def rewrite_file(path: Path, write: bool) -> FileResult:
    rel_path = path.relative_to(REPO_ROOT)
    result, new_text = rewrite_text(rel_path, read_text(path))
    if write and result.status == "updated" and new_text is not None:
        write_text(path, new_text)
    return result


def print_summary(results: list[FileResult]) -> None:
    counts = {
        "updated": 0,
        "unchanged": 0,
        "skipped": 0,
        "needs-manual-review": 0,
    }
    for result in results:
        counts[result.status] += 1

    print(f"checked: {len(results)}")
    print(f"updated: {counts['updated']}")
    print(f"unchanged: {counts['unchanged']}")
    print(f"skipped: {counts['skipped']}")
    print(f"needs-manual-review: {counts['needs-manual-review']}")

    manual = [result for result in results if result.status == "needs-manual-review"]
    if manual:
        print()
        print("manual review required:")
        for result in manual:
            print(f"- {result.path.as_posix()}: {result.note}")


def main() -> int:
    args = parse_args()
    results = [rewrite_file(path, write=args.write) for path in iter_target_files()]
    print_summary(results)

    has_manual = any(result.status == "needs-manual-review" for result in results)
    if has_manual:
        return 2
    if args.check and any(result.status == "updated" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
