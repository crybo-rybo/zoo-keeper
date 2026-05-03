#!/usr/bin/env python3
"""Compute CRAP (Change Risk Anti-Patterns) scores for zoo-keeper source files.

Formula: CRAP(m) = CC(m)^2 * (1 - cov(m)/100)^3 + CC(m)
  CC  = cyclomatic complexity per function (via lizard)
  cov = line coverage % within the function's line range (via gcovr)

A CRAP score > 30 (the original crap4j default) indicates a function that
is both complex and poorly tested — high change risk.
"""

import argparse
import csv
import datetime
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FunctionData:
    name: str
    file: Path
    start_line: int
    end_line: int
    ccn: int
    coverage_pct: float = 0.0
    crap_score: float = 0.0


def _require_tool(name: str) -> None:
    if subprocess.run(["which", name], capture_output=True).returncode != 0:
        print(f"error: '{name}' not found — install with: pip install {name}", file=sys.stderr)
        sys.exit(2)


def _detect_gcov_args() -> list[str]:
    """Return ['--gcov-executable', 'llvm-cov gcov'] for Clang, [] for GCC."""
    for exe in ("llvm-cov", "gcov"):
        if subprocess.run(["which", exe], capture_output=True).returncode == 0:
            return ["--gcov-executable", "llvm-cov gcov"] if exe == "llvm-cov" else []
    return []


def _run_lizard(src_dirs: list[Path]) -> list[FunctionData]:
    _require_tool("lizard")
    result = subprocess.run(
        ["lizard", "--csv"] + [str(d) for d in src_dirs],
        capture_output=True, text=True,
    )
    # lizard exits 1 when CCN threshold is exceeded; still valid output
    if result.returncode not in (0, 1):
        print(f"lizard error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    _LIZARD_FIELDS = [
        "NLOC", "CCN", "token", "PARAM", "length",
        "location", "file", "method_name", "method_long_name",
        "start_line", "end_line",
    ]
    functions: list[FunctionData] = []
    reader = csv.DictReader(io.StringIO(result.stdout), fieldnames=_LIZARD_FIELDS)
    for row in reader:
        try:
            # location field: "func_name@/abs/path/to/file.cpp:start_line"
            name = row["location"].split("@")[0] if "@" in row["location"] else row["location"]
            functions.append(FunctionData(
                name=name,
                file=Path(row["file"]).resolve(),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                ccn=int(row["CCN"]),
            ))
        except (KeyError, ValueError):
            continue
    return functions


def _run_gcovr(
    build_dir: Path,
    source_dir: Path,
    gcov_args: list[str],
) -> dict[Path, dict[int, int]]:
    """Returns {abs_file_path: {line_number: execution_count}}."""
    _require_tool("gcovr")
    filter_args = [
        "--filter", str(source_dir / "src") + "/",
        "--filter", str(source_dir / "include") + "/",
    ]
    cmd = [
        "gcovr",
        "--json", "-",
        "--root", str(source_dir),
        "--object-directory", str(build_dir),
    ] + gcov_args + filter_args

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"gcovr error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    coverage: dict[Path, dict[int, int]] = {}
    for entry in json.loads(result.stdout).get("files", []):
        # gcovr --root makes paths relative to source_dir
        # gcovr <7 uses "filename"; gcovr 7+ uses "file"
        fname = entry.get("filename") or entry.get("file")
        if not fname:
            continue
        fpath = (source_dir / fname).resolve()
        coverage[fpath] = {
            int(line["line_number"]): int(line["count"])
            for line in entry.get("lines", [])
            if line.get("count") is not None
        }
    return coverage


def _function_coverage(func: FunctionData, coverage: dict[Path, dict[int, int]]) -> float:
    line_map = coverage.get(func.file)
    if not line_map:
        return 0.0
    in_range = {ln: cnt for ln, cnt in line_map.items() if func.start_line <= ln <= func.end_line}
    if not in_range:
        return 0.0
    covered = sum(1 for cnt in in_range.values() if cnt > 0)
    return covered / len(in_range) * 100.0


def _crap(ccn: int, cov_pct: float) -> float:
    uncov = 1.0 - cov_pct / 100.0
    return ccn ** 2 * uncov ** 3 + ccn


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="CRAP score at which a function is flagged (default: 30)")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    build_dir = args.build_dir.resolve()

    src_dirs = [d for d in (source_dir / "src", source_dir / "include" / "zoo") if d.exists()]
    if not src_dirs:
        print(f"error: no src/ or include/zoo/ under {source_dir}", file=sys.stderr)
        return 1

    gcov_args = _detect_gcov_args()

    print("  [crap] running lizard...", file=sys.stderr)
    functions = _run_lizard(src_dirs)

    print("  [crap] running gcovr...", file=sys.stderr)
    coverage = _run_gcovr(build_dir, source_dir, gcov_args)

    for func in functions:
        func.coverage_pct = _function_coverage(func, coverage)
        func.crap_score = _crap(func.ccn, func.coverage_pct)

    functions.sort(key=lambda f: f.crap_score, reverse=True)
    over_threshold = [f for f in functions if f.crap_score > args.threshold]

    # --- report table ---
    W_NAME, W_FILE = 48, 34
    sep = "-" * (W_NAME + W_FILE + 30)
    print(f"\n{'Function':<{W_NAME}}  {'File':<{W_FILE}}  {'CC':>4}  {'Cov%':>6}  {'CRAP':>7}")
    print(sep)
    for func in functions:
        try:
            rel = func.file.relative_to(source_dir)
        except ValueError:
            rel = func.file
        flag = "  <-- over threshold" if func.crap_score > args.threshold else ""
        print(
            f"{func.name[:W_NAME]:<{W_NAME}}  {str(rel)[-W_FILE:]:<{W_FILE}}"
            f"  {func.ccn:>4}  {func.coverage_pct:>5.1f}%  {func.crap_score:>7.1f}{flag}"
        )

    print(f"\n  Total functions : {len(functions)}")
    print(f"  Threshold       : {args.threshold}")
    print(f"  Over threshold  : {len(over_threshold)}")

    report = [
        {
            "function": f.name,
            "file": str(f.file.relative_to(source_dir) if f.file.is_relative_to(source_dir) else f.file),
            "start_line": f.start_line,
            "end_line": f.end_line,
            "ccn": f.ccn,
            "coverage_pct": round(f.coverage_pct, 1),
            "crap": round(f.crap_score, 2),
        }
        for f in functions
    ]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_out = Path.cwd() / f"{timestamp}_crap_report.json"
    auto_out.write_text(json.dumps(report, indent=2))
    print(f"  JSON report     : {auto_out}")

    if over_threshold:
        print(
            f"\nFAIL: {len(over_threshold)} function(s) exceed CRAP threshold of {args.threshold}",
            file=sys.stderr,
        )
        return 1

    print(f"\nPASS: all {len(functions)} functions within CRAP threshold of {args.threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
