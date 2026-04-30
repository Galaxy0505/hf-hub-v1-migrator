from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import uuid
from pathlib import Path

from .ai_reviewer import AiConfig, add_ai_suggestions, load_env_file
from .report import FileResult, Summary
from .transformer import iter_python_files, transform_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hf-hub-v1-migrator",
        description="Migrate huggingface_hub v0.x Python call sites toward v1.x-safe APIs.",
    )
    parser.add_argument("paths", nargs="+", help="Python files or directories to scan.")
    parser.add_argument("--write", action="store_true", help="Write deterministic fixes to disk.")
    parser.add_argument("--diff", action="store_true", help="Print unified diffs for changed files.")
    parser.add_argument("--report", type=Path, help="Write a JSON report.")
    parser.add_argument(
        "--ai-fallback",
        action="store_true",
        help="Call an OpenAI-compatible chat endpoint for high-risk findings and store suggestions in the JSON report.",
    )
    parser.add_argument(
        "--apply-ai",
        action="store_true",
        help="Ask AI for structured JSON patches, run AI risk review, and apply only low-risk bounded local replacements.",
    )
    parser.add_argument(
        "--ai-risk-threshold",
        type=float,
        default=0.35,
        help="Maximum AI reviewer risk score allowed for --apply-ai. Defaults to 0.35.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Load AI configuration from this env file. Defaults to .env in the current directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ai_config = None
    if args.apply_ai:
        args.ai_fallback = True

    if args.ai_fallback:
        load_env_file(args.env_file)
        ai_config = AiConfig.from_env()
        if ai_config is None:
            print(
                "hf-hub-v1-migrator: --ai-fallback requested but HF_MIGRATOR_AI_BASE_URL "
                "or HF_MIGRATOR_AI_API_KEY is missing; report will contain review flags only.",
                file=sys.stderr,
            )

    files = list(iter_python_files(Path(path).resolve() for path in args.paths))
    results: list[FileResult] = []

    for file_path in files:
        source = file_path.read_text(encoding="utf-8")
        result = transform_source(source, path=str(file_path))
        findings = result.findings
        output_code = result.code
        ai_applied = 0

        if ai_config is not None:
            output_code, findings, ai_applied = add_ai_suggestions(
                output_code,
                findings,
                ai_config,
                apply_ai=args.apply_ai,
                risk_threshold=args.ai_risk_threshold,
            )

        changed = output_code != source

        if changed and args.diff:
            sys.stdout.write(
                "".join(
                    difflib.unified_diff(
                        source.splitlines(keepends=True),
                        output_code.splitlines(keepends=True),
                        fromfile=f"{file_path}:before",
                        tofile=f"{file_path}:after",
                    )
                )
            )

        if changed and args.write:
            atomic_write_text(file_path, output_code)

        results.append(
            FileResult(
                path=str(file_path),
                changed=changed,
                auto_fixes=result.auto_fixes + ai_applied,
                findings=findings,
            )
        )

    summary = Summary(
        files_scanned=len(results),
        files_changed=sum(1 for result in results if result.changed),
        auto_fixes=sum(result.auto_fixes for result in results),
        findings=sum(len(result.findings) for result in results),
    )

    payload = {
        "summary": summary.to_dict(),
        "ai_fallback": {
            "requested": bool(args.ai_fallback),
            "configured": ai_config is not None,
            "model": ai_config.model if ai_config is not None else None,
            "apply_requested": bool(args.apply_ai),
            "risk_threshold": args.ai_risk_threshold,
        },
        "files": [result.to_dict() for result in results],
    }

    if args.report:
        args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        "hf-hub-v1-migrator: "
        f"{summary.files_scanned} files scanned, "
        f"{summary.files_changed} files changed, "
        f"{summary.auto_fixes} deterministic fixes, "
        f"{summary.findings} findings."
    )

    return 0


def atomic_write_text(path: Path, content: str) -> None:
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
