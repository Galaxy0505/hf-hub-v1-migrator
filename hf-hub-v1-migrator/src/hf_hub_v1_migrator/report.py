from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    path: str
    line: int
    column: int
    message: str
    rule: str
    auto_fixed: bool = False
    ai_recommended: bool = False
    ai_suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FileResult:
    path: str
    changed: bool
    auto_fixes: int
    findings: list[Finding]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "changed": self.changed,
            "auto_fixes": self.auto_fixes,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class Summary:
    files_scanned: int
    files_changed: int
    auto_fixes: int
    findings: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
