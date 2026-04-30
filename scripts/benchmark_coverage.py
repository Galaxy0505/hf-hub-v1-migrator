from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=SyntaxWarning)

from hf_hub_v1_migrator.transformer import iter_python_files, transform_source  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the migrator in memory on real Python trees and report coverage/syntax safety."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Package roots to scan.")
    parser.add_argument("--output", type=Path, help="Write the benchmark report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_benchmark(args.paths)

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(payload)
    return 1 if payload["summary"]["syntax_failures"] else 0


def run_benchmark(paths: list[Path]) -> dict[str, Any]:
    packages: list[dict[str, Any]] = []
    total_codes: collections.Counter[str] = collections.Counter()
    summary = {
        "packages": 0,
        "files_scanned": 0,
        "files_changed": 0,
        "deterministic_fixes": 0,
        "review_findings": 0,
        "syntax_checked": 0,
        "syntax_failures": 0,
    }

    for root in paths:
        package = scan_package(root)
        packages.append(package)
        summary["packages"] += 1
        for key in (
            "files_scanned",
            "files_changed",
            "deterministic_fixes",
            "review_findings",
            "syntax_checked",
            "syntax_failures",
        ):
            summary[key] += package[key]
        total_codes.update(package["finding_codes"])

    return {
        "summary": summary,
        "finding_codes": dict(sorted(total_codes.items())),
        "packages": packages,
    }


def scan_package(root: Path) -> dict[str, Any]:
    files = list(iter_python_files([root.resolve()]))
    finding_codes: collections.Counter[str] = collections.Counter()
    failures: list[dict[str, str]] = []
    files_changed = 0
    deterministic_fixes = 0
    review_findings = 0
    syntax_checked = 0

    for file_path in files:
        try:
            source = file_path.read_text(encoding="utf-8")
            result = transform_source(source, path=str(file_path))
            compile(result.code, filename=str(file_path), mode="exec")
        except Exception as exc:  # pragma: no cover - diagnostic script
            failures.append({"path": str(file_path), "error": f"{type(exc).__name__}: {exc}"})
            continue

        syntax_checked += 1
        if result.changed:
            files_changed += 1
        deterministic_fixes += result.auto_fixes
        review_findings += sum(1 for finding in result.findings if finding.ai_recommended)
        finding_codes.update(finding.code for finding in result.findings)

    return {
        "path": str(root),
        "files_scanned": len(files),
        "files_changed": files_changed,
        "deterministic_fixes": deterministic_fixes,
        "review_findings": review_findings,
        "syntax_checked": syntax_checked,
        "syntax_failures": len(failures),
        "finding_codes": dict(sorted(finding_codes.items())),
        "failures": failures[:20],
    }


def print_summary(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print(
        "benchmark: "
        f"{summary['packages']} packages, "
        f"{summary['files_scanned']} files scanned, "
        f"{summary['files_changed']} files changed, "
        f"{summary['deterministic_fixes']} deterministic fixes, "
        f"{summary['review_findings']} review findings, "
        f"{summary['syntax_checked']} compiled, "
        f"{summary['syntax_failures']} syntax failures."
    )
    for package in payload["packages"]:
        print(
            f"- {package['path']}: "
            f"{package['files_changed']} changed, "
            f"{package['deterministic_fixes']} fixes, "
            f"{package['review_findings']} review, "
            f"{package['syntax_checked']}/{package['files_scanned']} compiled"
        )


if __name__ == "__main__":
    raise SystemExit(main())
