from __future__ import annotations

import json
import os
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import libcst as cst

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


def build_proposal_prompt(finding: Finding, code_context: str) -> str:
    return f"""You are an AI codemod proposal agent for migrating Python code to huggingface_hub v1.x.

Return ONLY valid JSON, with no markdown fences and no extra text.

Schema:
{{
  "action": "replace_block" | "manual_review",
  "target": {{
    "type": "function" | "statement",
    "line": {finding.line}
  }},
  "replacement": "Python code for the replacement block, or empty string for manual_review",
  "summary": "one sentence",
  "assumptions": ["short assumption"],
  "risk_factors": ["short risk factor"]
}}

Rules:
- Replace only the smallest enclosing function or statement block shown in the context.
- Preserve the original function signature when possible.
- Do not modify unrelated code.
- If the correct migration depends on unknown runtime behavior, choose manual_review.
- For Repository(...):
  - use snapshot_download for read/download intent
  - use HfApi().upload_file for a single-file upload
  - use HfApi().upload_folder for folder/model-directory push workflows

Finding:
- code: {finding.code}
- rule: {finding.rule}
- message: {finding.message}

Code context:
```python
{code_context}
```
"""


def build_review_prompt(finding: Finding, proposal: dict, code_context: str) -> str:
    return f"""You are an AI codemod risk reviewer. Review the proposed local patch for a huggingface_hub v1 migration.

Return ONLY valid JSON, with no markdown fences and no extra text.

Schema:
{{
  "decision": "apply" | "manual_review",
  "risk": 0.0,
  "reason": "short reason",
  "required_human_checks": ["short check"]
}}

Decision rules:
- Use "apply" only when the patch is local, syntax-safe, and behavior-preserving with low ambiguity.
- Use "manual_review" when behavior depends on runtime state, unknown repo type, side effects, or broad control-flow changes.
- Risk must be between 0 and 1.

Finding:
{json.dumps(finding.to_dict(), ensure_ascii=False)}

Proposal:
{json.dumps(proposal, ensure_ascii=False)}

Original context:
```python
{code_context}
```
"""


def add_ai_suggestions(
    source: str,
    findings: list[Finding],
    config: AiConfig,
    *,
    apply_ai: bool = False,
    risk_threshold: float = 0.35,
) -> tuple[str, list[Finding], int]:
    code = source
    updated: list[Finding] = []
    applied = 0
    for finding in findings:
        if not finding.ai_recommended:
            updated.append(finding)
            continue
        context = extract_context(code, finding.line)
        prompt = build_finding_prompt(finding, context)
        try:
            suggestion = call_openai_compatible(config, prompt)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            suggestion = f"AI request failed: {exc}"

        proposal = None
        review = None
        ai_applied = False
        apply_reason = None

        if apply_ai:
            try:
                proposal = request_json(config, build_proposal_prompt(finding, context))
                review = request_json(config, build_review_prompt(finding, proposal, context))
                code, ai_applied, apply_reason = maybe_apply_proposal(
                    code,
                    finding,
                    proposal,
                    review,
                    risk_threshold=risk_threshold,
                )
                if ai_applied:
                    applied += 1
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, cst.ParserSyntaxError) as exc:
                apply_reason = f"AI apply skipped: {exc}"

        updated.append(
            replace(
                finding,
                ai_suggestion=suggestion,
                ai_proposal=proposal,
                ai_review=review,
                ai_applied=ai_applied,
                ai_apply_reason=apply_reason,
            )
        )
    return code, updated, applied


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


def request_json(config: AiConfig, prompt: str) -> dict:
    raw = call_openai_compatible(config, prompt)
    cleaned = strip_json_fence(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("AI response JSON must be an object")
    return data


def strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def maybe_apply_proposal(
    source: str,
    finding: Finding,
    proposal: dict,
    review: dict,
    *,
    risk_threshold: float,
) -> tuple[str, bool, str]:
    decision = review.get("decision")
    risk = review.get("risk")
    if decision != "apply":
        return source, False, "review decision requires manual review"
    if not isinstance(risk, (int, float)) or float(risk) > risk_threshold:
        return source, False, f"risk {risk!r} exceeds threshold {risk_threshold}"
    if proposal.get("action") != "replace_block":
        return source, False, "proposal is not replace_block"
    replacement = proposal.get("replacement")
    if not isinstance(replacement, str) or not replacement.strip():
        return source, False, "proposal has no replacement code"

    target = proposal.get("target") if isinstance(proposal.get("target"), dict) else {}
    target_type = target.get("type")
    if target_type not in {"function", "statement"}:
        return source, False, "unsupported proposal target type"

    new_source = replace_local_block(source, finding.line, replacement, target_type=target_type)
    cst.parse_module(new_source)
    compile(new_source, filename=finding.path, mode="exec")
    return new_source, True, "applied bounded AI patch"


def replace_local_block(source: str, line: int, replacement: str, *, target_type: str) -> str:
    module = cst.parse_module(source)
    locator = BlockLocator(line=line, target_type=target_type)
    wrapper = cst.metadata.MetadataWrapper(module)
    wrapper.visit(locator)
    if locator.range is None:
        raise ValueError(f"Could not locate enclosing {target_type} block for line {line}")

    start = locator.range.start.line
    end = locator.range.end.line
    original_lines = source.splitlines(keepends=True)
    newline = "\r\n" if any(part.endswith("\r\n") for part in original_lines) else "\n"
    replacement_lines = replacement.strip("\n").splitlines()
    replacement_text = newline.join(replacement_lines) + newline
    return "".join(original_lines[: start - 1]) + replacement_text + "".join(original_lines[end:])


class BlockLocator(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, *, line: int, target_type: str) -> None:
        self.line = line
        self.target_type = target_type
        self.range: cst.metadata.CodeRange | None = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if self.target_type != "function":
            return None
        code_range = self.get_metadata(cst.metadata.PositionProvider, node)
        if code_range.start.line <= self.line <= code_range.end.line:
            if self.range is None or span_size(code_range) < span_size(self.range):
                self.range = code_range
        return None

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> bool | None:
        if self.target_type != "statement":
            return None
        code_range = self.get_metadata(cst.metadata.PositionProvider, node)
        if code_range.start.line <= self.line <= code_range.end.line:
            self.range = code_range
        return None


def span_size(code_range: cst.metadata.CodeRange) -> int:
    return code_range.end.line - code_range.start.line


def chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"
