from __future__ import annotations

import json
import os
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .report import Finding


@dataclass(frozen=True)
class AiReviewRequest:
    path: str
    line: int
    symbol: str
    source: str
    reason: str


@dataclass(frozen=True)
class AiConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60

    @classmethod
    def from_env(cls) -> "AiConfig | None":
        base_url = os.environ.get("HF_MIGRATOR_AI_BASE_URL")
        api_key = os.environ.get("HF_MIGRATOR_AI_API_KEY")
        model = os.environ.get("HF_MIGRATOR_AI_MODEL", "gpt-4o-mini")
        if not base_url or not api_key:
            return None
        return cls(base_url=base_url, api_key=api_key, model=model)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def build_repository_prompt(request: AiReviewRequest) -> str:
    """Build the bounded prompt used by the optional AI fallback.

    The MVP keeps AI suggestions out of the deterministic rewrite path. The CLI
    records these prompts in the report so they can be reviewed or wired to an
    LLM provider without changing the codemod's safety model.
    """

    return f"""You are migrating huggingface_hub code to v1.x.

File: {request.path}
Line: {request.line}
Legacy symbol: {request.symbol}
Reason: {request.reason}

The old Repository class is git-backed. Prefer v1.x HTTP APIs:
- snapshot_download for download/sync reads
- HfApi().upload_file for single-file uploads
- HfApi().upload_folder for folder uploads
- HfApi methods for repo creation and metadata changes

Rewrite only the function or statement block below. Preserve behavior and avoid
unrelated edits. If behavior is ambiguous, return a short TODO comment instead
of guessing.

```python
{request.source}
```
"""


def build_finding_prompt(finding: Finding, code_context: str) -> str:
    return f"""You are assisting a deterministic Python codemod that migrates huggingface_hub v0.x code to v1.x.

Finding:
- code: {finding.code}
- rule: {finding.rule}
- file: {finding.path}
- line: {finding.line}
- message: {finding.message}

Return a concise migration suggestion. If the fix is safe, include a minimal Python replacement snippet.
If behavior is ambiguous, say what the developer must inspect instead of guessing.

Relevant code context:

```python
{code_context}
```
"""


def add_ai_suggestions(source: str, findings: list[Finding], config: AiConfig) -> list[Finding]:
    updated: list[Finding] = []
    for finding in findings:
        if not finding.ai_recommended:
            updated.append(finding)
            continue
        prompt = build_finding_prompt(finding, extract_context(source, finding.line))
        try:
            suggestion = call_openai_compatible(config, prompt)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            suggestion = f"AI request failed: {exc}"
        updated.append(replace(finding, ai_suggestion=suggestion))
    return updated


def extract_context(source: str, line: int, radius: int = 25) -> str:
    lines = source.splitlines()
    start = max(line - radius - 1, 0)
    end = min(line + radius, len(lines))
    numbered = []
    for index in range(start, end):
        numbered.append(f"{index + 1:>4}: {lines[index]}")
    return "\n".join(numbered)


def call_openai_compatible(config: AiConfig, prompt: str) -> str:
    endpoint = chat_completions_url(config.base_url)
    payload = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You are a senior Python migration engineer. Be conservative and do not invent behavior.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=config.timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected AI response shape: {data}") from exc


def chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"
