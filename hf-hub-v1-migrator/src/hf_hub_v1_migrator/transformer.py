from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst
import libcst.matchers as m
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider, ScopeProvider

from .report import Finding


HF_MODULE = "huggingface_hub"

RENAMED_SYMBOLS = {
    "InferenceApi": "InferenceClient",
    "update_repo_visibility": "update_repo_settings",
}

DOWNLOAD_FUNCTIONS = {
    f"{HF_MODULE}.hf_hub_download",
    f"{HF_MODULE}.snapshot_download",
    f"{HF_MODULE}.cached_download",
}

REMOVED_DOWNLOAD_KWS = {
    "resume_download",
    "force_filename",
    "local_dir_use_symlinks",
}

LIST_MODELS_REVIEW_KWS = {
    "cardData",
    "fetch_config",
    "full",
    "library",
    "language",
    "tags",
    "task",
}

DYNAMIC_KWARGS_REVIEW_TARGETS = DOWNLOAD_FUNCTIONS | {
    f"{HF_MODULE}.login",
    f"{HF_MODULE}.list_models",
    f"{HF_MODULE}.model_info",
    f"{HF_MODULE}.dataset_info",
    f"{HF_MODULE}.space_info",
    f"{HF_MODULE}.Repository",
    f"{HF_MODULE}.InferenceApi",
    f"{HF_MODULE}.InferenceClient",
    f"{HF_MODULE}.configure_http_backend",
    f"{HF_MODULE}.update_repo_visibility",
    f"{HF_MODULE}.update_repo_settings",
}


@dataclass(frozen=True)
class MigrationResult:
    code: str
    changed: bool
    auto_fixes: int
    findings: list[Finding]


@dataclass(frozen=True)
class ImportState:
    hf_module_aliases: set[str]
    hf_symbols: dict[str, str]
    requests_module_aliases: set[str]
    requests_http_error_aliases: set[str]
    has_hf_usage: bool


def dotted_name(node: cst.CSTNode | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        prefix = dotted_name(node.value)
        if prefix is None:
            return None
        return f"{prefix}.{node.attr.value}"
    return None


def local_import_name(alias: cst.ImportAlias) -> str | None:
    if alias.asname is not None:
        return alias.asname.name.value
    name = dotted_name(alias.name)
    if name is None:
        return None
    return name.split(".")[0]


def imported_symbol_name(alias: cst.ImportAlias) -> str | None:
    name = dotted_name(alias.name)
    if name is None:
        return None
    return name.split(".")[-1]


def normalize_target(raw: str | None, imports: ImportState) -> str | None:
    if raw is None:
        return None
    parts = raw.split(".")
    if not parts:
        return None

    head = parts[0]
    tail = ".".join(parts[1:])
    if head in imports.hf_module_aliases:
        return f"{HF_MODULE}.{tail}" if tail else HF_MODULE

    imported = imports.hf_symbols.get(head)
    if imported is not None:
        return f"{HF_MODULE}.{imported}.{tail}" if tail else f"{HF_MODULE}.{imported}"

    return raw


def is_true(node: cst.BaseExpression) -> bool:
    return m.matches(node, m.Name("True"))


def is_false(node: cst.BaseExpression) -> bool:
    return m.matches(node, m.Name("False"))


class ImportCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.hf_module_aliases: set[str] = set()
        self.hf_symbols: dict[str, str] = {}
        self.requests_module_aliases: set[str] = set()
        self.requests_http_error_aliases: set[str] = set()
        self.has_hf_usage = False

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            name = dotted_name(alias.name)
            local = local_import_name(alias)
            if name == HF_MODULE and local is not None:
                self.hf_module_aliases.add(local)
                self.has_hf_usage = True
            if name == "requests" and local is not None:
                self.requests_module_aliases.add(local)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        module_name = dotted_name(node.module)
        if not isinstance(node.names, tuple):
            return

        if module_name == HF_MODULE:
            self.has_hf_usage = True
            for alias in node.names:
                imported = imported_symbol_name(alias)
                local = local_import_name(alias)
                if imported is not None and local is not None:
                    self.hf_symbols[local] = imported
            return

        if module_name is not None and module_name.startswith(f"{HF_MODULE}."):
            self.has_hf_usage = True
            return

        if module_name == "requests":
            for alias in node.names:
                imported = imported_symbol_name(alias)
                local = local_import_name(alias)
                if imported == "HTTPError" and local is not None:
                    self.requests_http_error_aliases.add(local)

    def state(self) -> ImportState:
        return ImportState(
            hf_module_aliases=self.hf_module_aliases,
            hf_symbols=self.hf_symbols,
            requests_module_aliases=self.requests_module_aliases,
            requests_http_error_aliases=self.requests_http_error_aliases,
            has_hf_usage=self.has_hf_usage,
        )


class HfCallDetector(cst.CSTVisitor):
    def __init__(self, imports: ImportState) -> None:
        self.imports = imports
        self.found = False

    def visit_Call(self, node: cst.Call) -> bool | None:
        target = normalize_target(dotted_name(node.func), self.imports)
        if target is not None and target.startswith(f"{HF_MODULE}."):
            self.found = True
            return False
        return None


class HuggingFaceHubTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)

    def __init__(self, path: str, imports: ImportState) -> None:
        self.path = path
        self.imports = imports
        self.findings: list[Finding] = []
        self.auto_fixes = 0
        self.needed_hf_imports: set[str] = set()

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        module_name = dotted_name(original_node.module)
        if module_name != HF_MODULE or not isinstance(updated_node.names, tuple):
            return updated_node

        changed = False
        new_aliases: list[cst.ImportAlias] = []
        for original_alias, alias in zip(original_node.names, updated_node.names):
            imported = imported_symbol_name(original_alias)
            replacement = RENAMED_SYMBOLS.get(imported or "")
            if replacement is not None and isinstance(alias.name, cst.Name):
                changed = True
                new_aliases.append(alias.with_changes(name=cst.Name(replacement)))
                self.record(
                    original_alias,
                    code="HF101",
                    severity="info",
                    rule="rename-import",
                    message=f"Renamed huggingface_hub.{imported} import to {replacement}.",
                    auto_fixed=True,
                )
            else:
                new_aliases.append(alias)

        if not changed:
            return updated_node
        return updated_node.with_changes(names=tuple(new_aliases))

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        raw_target = dotted_name(original_node.func)
        target = normalize_target(raw_target, self.imports)
        if target is None or not target.startswith(f"{HF_MODULE}."):
            return updated_node

        updated = updated_node
        updated = self.rewrite_call_target(original_node, updated, target)
        updated = self.rewrite_call_args(original_node, updated, target)
        self.flag_high_risk_calls(original_node, target)
        return updated

    def leave_Try(self, original_node: cst.Try, updated_node: cst.Try) -> cst.Try:
        detector = HfCallDetector(self.imports)
        original_node.body.visit(detector)

        new_handlers: list[cst.ExceptHandler] = []
        changed = False
        for original_handler, handler in zip(original_node.handlers, updated_node.handlers):
            if original_handler.type is None or not self.is_requests_http_error(original_handler.type):
                new_handlers.append(handler)
                continue

            if detector.found:
                self.needed_hf_imports.add("HfHubHttpError")
                self.auto_fixes += 1
                changed = True
                new_handlers.append(handler.with_changes(type=cst.Name("HfHubHttpError")))
                self.record(
                    original_handler,
                    code="HF301",
                    severity="info",
                    rule="http-error-exception",
                    message="Replaced requests.HTTPError with HfHubHttpError for Hugging Face API error handling.",
                    auto_fixed=True,
                )
                continue

            if self.imports.has_hf_usage:
                self.record(
                    original_handler,
                    code="HF302",
                    severity="warning",
                    rule="http-error-exception-review",
                    message="requests.HTTPError is used in a module that also imports Hugging Face Hub; review whether HfHubHttpError is required.",
                    ai_recommended=True,
                )
            new_handlers.append(handler)

        if not changed:
            return updated_node
        return updated_node.with_changes(handlers=tuple(new_handlers))

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if not self.needed_hf_imports:
            return updated_node
        return ensure_hf_imports(updated_node, self.needed_hf_imports)

    def rewrite_call_target(
        self, original_node: cst.Call, updated_node: cst.Call, target: str
    ) -> cst.Call:
        if target == f"{HF_MODULE}.InferenceApi":
            new_func = replace_leaf_attr_or_name(
                original_node.func,
                updated_node.func,
                old_name="InferenceApi",
                new_name="InferenceClient",
            )
            if new_func is not updated_node.func:
                self.auto_fixes += 1
                self.record(
                    original_node.func,
                    code="HF102",
                    severity="info",
                    rule="inference-api",
                    message="Replaced deprecated InferenceApi call with InferenceClient.",
                    auto_fixed=True,
                )
                return updated_node.with_changes(func=new_func)

        if target == f"{HF_MODULE}.update_repo_visibility":
            new_func = replace_leaf_attr_or_name(
                original_node.func,
                updated_node.func,
                old_name="update_repo_visibility",
                new_name="update_repo_settings",
            )
            if new_func is not updated_node.func:
                self.auto_fixes += 1
                self.record(
                    original_node.func,
                    code="HF103",
                    severity="info",
                    rule="update-repo-visibility",
                    message="Replaced update_repo_visibility with update_repo_settings.",
                    auto_fixed=True,
                )
                return updated_node.with_changes(func=new_func)

        if target == f"{HF_MODULE}.HfFolder.get_token":
            if isinstance(original_node.func, cst.Attribute):
                if isinstance(original_node.func.value, cst.Attribute):
                    new_value = updated_node.func.value.value  # hf.HfFolder -> hf
                    new_func = cst.Attribute(value=new_value, attr=cst.Name("get_token"))
                else:
                    self.needed_hf_imports.add("get_token")
                    new_func = cst.Name("get_token")

                self.auto_fixes += 1
                self.record(
                    original_node.func,
                    code="HF104",
                    severity="info",
                    rule="hffolder-get-token",
                    message="Replaced HfFolder.get_token() with get_token().",
                    auto_fixed=True,
                )
                return updated_node.with_changes(func=new_func)

        return updated_node

    def rewrite_call_args(
        self, original_node: cst.Call, updated_node: cst.Call, target: str
    ) -> cst.Call:
        new_args: list[cst.Arg] = []
        changed = False

        for original_arg, arg in zip(original_node.args, updated_node.args):
            if original_arg.star == "**" and target in DYNAMIC_KWARGS_REVIEW_TARGETS:
                self.record(
                    original_arg,
                    code="HF401",
                    severity="warning",
                    rule="dynamic-kwargs-review",
                    message="Dynamic **kwargs passed to a Hugging Face Hub call; inspect for v1.x removed keys such as use_auth_token or resume_download.",
                    ai_recommended=True,
                )
                new_args.append(arg)
                continue

            keyword = original_arg.keyword.value if original_arg.keyword is not None else None

            if keyword == "use_auth_token":
                changed = True
                self.auto_fixes += 1
                new_args.append(arg.with_changes(keyword=cst.Name("token")))
                self.record(
                    original_arg,
                    code="HF201",
                    severity="info",
                    rule="use-auth-token",
                    message="Renamed use_auth_token= to token=.",
                    auto_fixed=True,
                )
                continue

            if target in DOWNLOAD_FUNCTIONS and keyword in REMOVED_DOWNLOAD_KWS:
                changed = True
                self.auto_fixes += 1
                self.record(
                    original_arg,
                    code="HF202",
                    severity="info",
                    rule="removed-download-kw",
                    message=f"Removed deprecated {keyword}= argument from {target.split('.')[-1]}().",
                    auto_fixed=True,
                )
                continue

            if target == f"{HF_MODULE}.list_models" and keyword in LIST_MODELS_REVIEW_KWS:
                self.record(
                    original_arg,
                    code="HF402",
                    severity="warning",
                    rule="list-models-filter-review",
                    message=f"list_models({keyword}=...) changed in huggingface_hub v1.x; review manually instead of broadening results automatically.",
                    ai_recommended=True,
                )

            if target == f"{HF_MODULE}.login" and keyword == "write_permission":
                changed = True
                self.auto_fixes += 1
                self.record(
                    original_arg,
                    code="HF203",
                    severity="info",
                    rule="login-write-permission",
                    message="Removed deprecated login(write_permission=...) argument.",
                    auto_fixed=True,
                )
                continue

            if target == f"{HF_MODULE}.login" and keyword == "new_session":
                if is_true(original_arg.value):
                    changed = True
                    self.auto_fixes += 1
                    new_args.append(arg.with_changes(keyword=cst.Name("skip_if_logged_in"), value=cst.Name("False")))
                    self.record(
                        original_arg,
                        code="HF204",
                        severity="info",
                        rule="login-new-session",
                        message="Replaced login(new_session=True) with skip_if_logged_in=False.",
                        auto_fixed=True,
                    )
                    continue
                if is_false(original_arg.value):
                    changed = True
                    self.auto_fixes += 1
                    new_args.append(arg.with_changes(keyword=cst.Name("skip_if_logged_in"), value=cst.Name("True")))
                    self.record(
                        original_arg,
                        code="HF204",
                        severity="info",
                        rule="login-new-session",
                        message="Replaced login(new_session=False) with skip_if_logged_in=True.",
                        auto_fixed=True,
                    )
                    continue
                self.record(
                    original_arg,
                    code="HF403",
                    severity="warning",
                    rule="login-new-session-review",
                    message="Dynamic login(new_session=...) requires manual conversion to skip_if_logged_in=not new_session.",
                    ai_recommended=True,
                )

            new_args.append(arg)

        if not changed:
            return updated_node
        return updated_node.with_changes(
            args=tuple(normalize_last_arg_comma(new_args, multiline=call_has_multiline_args(original_node)))
        )

    def flag_high_risk_calls(self, original_node: cst.Call, target: str) -> None:
        if target == f"{HF_MODULE}.Repository":
            self.record(
                original_node,
                code="HF501",
                severity="warning",
                rule="repository-ai-review",
                message="Repository is git-backed and removed in v1.x; rewrite to snapshot_download, HfApi.upload_file, or HfApi.upload_folder based on surrounding logic.",
                ai_recommended=True,
            )

        if target == f"{HF_MODULE}.configure_http_backend":
            self.record(
                original_node,
                code="HF502",
                severity="warning",
                rule="http-backend-ai-review",
                message="configure_http_backend was replaced by HTTPX client factories; review custom backend logic manually.",
                ai_recommended=True,
            )

        if target.startswith(f"{HF_MODULE}.HfFolder.") and not target.endswith(".get_token"):
            self.record(
                original_node,
                code="HF503",
                severity="warning",
                rule="hffolder-ai-review",
                message="HfFolder methods other than get_token need semantic migration to login/logout/token helpers.",
                ai_recommended=True,
            )

    def is_requests_http_error(self, node: cst.BaseExpression) -> bool:
        raw = dotted_name(node)
        if raw is None:
            return False
        parts = raw.split(".")
        if len(parts) == 2 and parts[1] == "HTTPError":
            return parts[0] in self.imports.requests_module_aliases
        return raw in self.imports.requests_http_error_aliases

    def record(
        self,
        node: cst.CSTNode,
        *,
        code: str,
        severity: str,
        rule: str,
        message: str,
        auto_fixed: bool = False,
        ai_recommended: bool = False,
    ) -> None:
        position = self.node_position(node)
        self.findings.append(
            Finding(
                code=code,
                severity=severity,  # type: ignore[arg-type]
                path=self.path,
                line=position.start.line,
                column=position.start.column,
                message=message,
                rule=rule,
                auto_fixed=auto_fixed,
                ai_recommended=ai_recommended,
            )
        )

    def node_position(self, node: cst.CSTNode) -> CodeRange:
        try:
            return self.get_metadata(PositionProvider, node)
        except KeyError:
            return CodeRange((0, 0), (0, 0))


def replace_leaf_attr_or_name(
    original_func: cst.BaseExpression,
    updated_func: cst.BaseExpression,
    *,
    old_name: str,
    new_name: str,
) -> cst.BaseExpression:
    if isinstance(original_func, cst.Name) and original_func.value == old_name:
        return cst.Name(new_name)
    if isinstance(original_func, cst.Attribute) and original_func.attr.value == old_name:
        return updated_func.with_changes(attr=cst.Name(new_name))
    return updated_func


def call_has_multiline_args(call: cst.Call) -> bool:
    return any(
        isinstance(arg.comma, cst.Comma)
        and isinstance(arg.comma.whitespace_after, cst.ParenthesizedWhitespace)
        for arg in call.args
    )


def normalize_last_arg_comma(args: list[cst.Arg], *, multiline: bool) -> list[cst.Arg]:
    if not args:
        return args
    args = list(args)
    if multiline:
        args[-1] = args[-1].with_changes(
            comma=cst.Comma(whitespace_after=cst.ParenthesizedWhitespace(indent=True))
        )
    else:
        args[-1] = args[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
    return args


def ensure_hf_imports(module: cst.Module, imports: Iterable[str]) -> cst.Module:
    needed = set(imports)
    if not needed:
        return module

    body = list(module.body)
    for index, statement in enumerate(body):
        import_from = extract_single_import_from(statement)
        if import_from is None or dotted_name(import_from.module) != HF_MODULE:
            continue
        if not isinstance(import_from.names, tuple):
            continue

        local_names = {
            alias.asname.name.value if alias.asname is not None else dotted_name(alias.name)
            for alias in import_from.names
        }
        missing = sorted(name for name in needed if name not in local_names)
        if not missing:
            return module

        aliases = list(import_from.names)
        aliases.extend(cst.ImportAlias(name=cst.Name(name)) for name in missing)
        new_import = import_from.with_changes(names=tuple(aliases))
        body[index] = statement.with_changes(body=[new_import])  # type: ignore[arg-type]
        return module.with_changes(body=tuple(body))

    import_statement = cst.SimpleStatementLine(
        body=[
            cst.ImportFrom(
                module=cst.Name(HF_MODULE),
                names=tuple(cst.ImportAlias(name=cst.Name(name)) for name in sorted(needed)),
            )
        ]
    )
    insert_at = first_import_insert_index(body)
    body.insert(insert_at, import_statement)
    return module.with_changes(body=tuple(body))


def extract_single_import_from(statement: cst.BaseStatement) -> cst.ImportFrom | None:
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return None
    small = statement.body[0]
    if isinstance(small, cst.ImportFrom):
        return small
    return None


def first_import_insert_index(body: list[cst.BaseStatement]) -> int:
    index = 0
    if body and is_module_docstring(body[0]):
        index = 1
    while index < len(body) and is_future_import(body[index]):
        index += 1
    return index


def is_module_docstring(statement: cst.BaseStatement) -> bool:
    return (
        isinstance(statement, cst.SimpleStatementLine)
        and len(statement.body) == 1
        and isinstance(statement.body[0], cst.Expr)
        and isinstance(statement.body[0].value, cst.SimpleString)
    )


def is_future_import(statement: cst.BaseStatement) -> bool:
    import_from = extract_single_import_from(statement)
    return import_from is not None and dotted_name(import_from.module) == "__future__"


def transform_source(source: str, path: str = "<string>") -> MigrationResult:
    module = cst.parse_module(source)

    collector = ImportCollector()
    module.visit(collector)

    wrapper = MetadataWrapper(module)
    transformer = HuggingFaceHubTransformer(path=path, imports=collector.state())
    updated = wrapper.visit(transformer)
    code = updated.code

    return MigrationResult(
        code=code,
        changed=code != source,
        auto_fixes=transformer.auto_fixes,
        findings=transformer.findings,
    )


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    ignored_dirs = {".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "venv", "__pycache__"}
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
            continue
        if path.is_dir():
            for child in path.rglob("*.py"):
                if ignored_dirs.intersection(child.parts):
                    continue
                yield child
