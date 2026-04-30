from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst
import libcst.matchers as m
from libcst.metadata import CodeRange, MetadataWrapper, ParentNodeProvider, PositionProvider

from .report import Finding


HF_MODULE = "huggingface_hub"

RENAMED_SYMBOLS = {
    "InferenceApi": "InferenceClient",
    "update_repo_visibility": "update_repo_settings",
}

CLEANUP_HF_IMPORTS_IF_UNUSED = {
    "HfFolder",
    "InferenceApi",
    "cached_download",
    "hf_hub_url",
}

DOWNLOAD_FUNCTIONS = {
    f"{HF_MODULE}.hf_hub_download",
    f"{HF_MODULE}.snapshot_download",
    f"{HF_MODULE}.cached_download",
    f"{HF_MODULE}.file_download.hf_hub_download",
    f"{HF_MODULE}.file_download.snapshot_download",
    f"{HF_MODULE}.file_download.cached_download",
}

REMOVED_DOWNLOAD_KWS = {
    "resume_download",
    "force_filename",
    "local_dir_use_symlinks",
}

REMOVED_LIST_MODELS_FILTER_KWS = {
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

HFAPI_DYNAMIC_KWARGS_REVIEW_METHODS = {
    "list_models",
    "model_info",
    "dataset_info",
    "space_info",
    "update_repo_visibility",
    "update_repo_settings",
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
    shadowed_hf_cache_home: bool


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


def import_replacement(
    module_name: str | None, imported: str | None, *, allow_constant_rewrite: bool = True
) -> str | None:
    if module_name == HF_MODULE:
        return RENAMED_SYMBOLS.get(imported or "")
    if allow_constant_rewrite and module_name == f"{HF_MODULE}.constants" and imported == "hf_cache_home":
        return "HF_HOME"
    return None


def normalize_hfapi_instance_target(
    normalized_target: str | None, raw_target: str | None, hfapi_names: set[str]
) -> str | None:
    if raw_target is None:
        return normalized_target
    parts = raw_target.split(".", 1)
    if len(parts) == 2 and parts[0] in hfapi_names:
        return f"{HF_MODULE}.HfApi.{parts[1]}"
    return normalized_target


def is_hfapi_constructor_target(target: str | None) -> bool:
    return target is not None and target.startswith(f"{HF_MODULE}.") and target.endswith(".HfApi")


def is_list_models_target(target: str) -> bool:
    return target in {f"{HF_MODULE}.list_models", f"{HF_MODULE}.HfApi.list_models"}


def is_build_hf_headers_target(target: str) -> bool:
    return target in {f"{HF_MODULE}.build_hf_headers", f"{HF_MODULE}.utils.build_hf_headers"}


def is_cached_download_target(target: str) -> bool:
    return target in {f"{HF_MODULE}.cached_download", f"{HF_MODULE}.file_download.cached_download"}


def is_hf_hub_url_target(target: str | None) -> bool:
    return target in {f"{HF_MODULE}.hf_hub_url", f"{HF_MODULE}.file_download.hf_hub_url"}


def is_dynamic_kwargs_review_target(target: str) -> bool:
    if target in DYNAMIC_KWARGS_REVIEW_TARGETS:
        return True
    prefix = f"{HF_MODULE}.HfApi."
    return target.startswith(prefix) and target.removeprefix(prefix) in HFAPI_DYNAMIC_KWARGS_REVIEW_METHODS


def safe_list_models_filter_arg(call: cst.Call, target: str) -> str | None:
    if not is_list_models_target(target):
        return None
    if any(arg.keyword is not None and arg.keyword.value == "filter" for arg in call.args):
        return None

    removed_keywords = [
        arg.keyword.value
        for arg in call.args
        if arg.keyword is not None and arg.keyword.value in REMOVED_LIST_MODELS_FILTER_KWS
    ]
    if len(removed_keywords) == 1:
        return removed_keywords[0]
    return None


def is_true(node: cst.BaseExpression) -> bool:
    return m.matches(node, m.Name("True"))


def is_false(node: cst.BaseExpression) -> bool:
    return m.matches(node, m.Name("False"))


def function_param_names(params: cst.Parameters) -> set[str]:
    names = {param.name.value for param in params.posonly_params}
    names.update(param.name.value for param in params.params)
    names.update(param.name.value for param in params.kwonly_params)
    if isinstance(params.star_arg, cst.Param):
        names.add(params.star_arg.name.value)
    if params.star_kwarg is not None:
        names.add(params.star_kwarg.name.value)
    return names


class ImportCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.hf_module_aliases: set[str] = set()
        self.hf_symbols: dict[str, str] = {}
        self.requests_module_aliases: set[str] = set()
        self.requests_http_error_aliases: set[str] = set()
        self.has_hf_usage = False
        self.shadowed_hf_cache_home = False

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
            submodule = module_name[len(HF_MODULE) + 1 :]
            for alias in node.names:
                imported = imported_symbol_name(alias)
                local = local_import_name(alias)
                if imported is not None and local is not None:
                    self.hf_symbols[local] = f"{submodule}.{imported}"
            return

        if module_name == "requests":
            for alias in node.names:
                imported = imported_symbol_name(alias)
                local = local_import_name(alias)
                if imported == "HTTPError" and local is not None:
                    self.requests_http_error_aliases.add(local)

        if module_name == "requests.exceptions":
            for alias in node.names:
                imported = imported_symbol_name(alias)
                local = local_import_name(alias)
                if imported == "HTTPError" and local is not None:
                    self.requests_http_error_aliases.add(local)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        if node.name.value == "hf_cache_home" or "hf_cache_home" in function_param_names(node.params):
            self.shadowed_hf_cache_home = True

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        if node.name.value == "hf_cache_home":
            self.shadowed_hf_cache_home = True

    def visit_AssignTarget(self, node: cst.AssignTarget) -> None:
        if isinstance(node.target, cst.Name) and node.target.value == "hf_cache_home":
            self.shadowed_hf_cache_home = True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if isinstance(node.target, cst.Name) and node.target.value == "hf_cache_home":
            self.shadowed_hf_cache_home = True

    def state(self) -> ImportState:
        return ImportState(
            hf_module_aliases=self.hf_module_aliases,
            hf_symbols=self.hf_symbols,
            requests_module_aliases=self.requests_module_aliases,
            requests_http_error_aliases=self.requests_http_error_aliases,
            has_hf_usage=self.has_hf_usage,
            shadowed_hf_cache_home=self.shadowed_hf_cache_home,
        )


class HfCallDetector(cst.CSTVisitor):
    def __init__(self, imports: ImportState, hfapi_names: set[str] | None = None) -> None:
        self.imports = imports
        self.hfapi_names = set(hfapi_names or set())
        self.found = False

    def visit_Call(self, node: cst.Call) -> bool | None:
        target = normalize_hfapi_instance_target(
            normalize_target(dotted_name(node.func), self.imports),
            dotted_name(node.func),
            self.hfapi_names,
        )
        if target is not None and target.startswith(f"{HF_MODULE}."):
            self.found = True
            return False
        return None


class HuggingFaceHubTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider)

    def __init__(self, path: str, imports: ImportState) -> None:
        self.path = path
        self.imports = imports
        self.findings: list[Finding] = []
        self.auto_fixes = 0
        self.needed_hf_imports: set[str] = set()
        self.hfapi_scopes: list[set[str]] = [set()]
        self.shadow_scopes: list[set[str]] = [set()]
        self.import_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.hfapi_scopes.append(set())
        self.shadow_scopes.append(function_param_names(node.params))
        return None

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        self.hfapi_scopes.pop()
        self.shadow_scopes.pop()
        return updated_node

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        for target in original_node.targets:
            self.track_hfapi_assignment(target.target, original_node.value)
            self.track_shadow_assignment(target.target)
        return updated_node

    def leave_AnnAssign(
        self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign
    ) -> cst.AnnAssign:
        if original_node.value is not None:
            self.track_hfapi_assignment(original_node.target, original_node.value)
        self.track_shadow_assignment(original_node.target)
        return updated_node

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool | None:
        self.import_depth += 1
        return None

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        self.import_depth -= 1
        module_name = dotted_name(original_node.module)
        if module_name not in {HF_MODULE, f"{HF_MODULE}.constants"} or not isinstance(updated_node.names, tuple):
            return updated_node

        changed = False
        new_aliases: list[cst.ImportAlias] = []
        for original_alias, alias in zip(original_node.names, updated_node.names):
            imported = imported_symbol_name(original_alias)
            replacement = import_replacement(
                module_name,
                imported,
                allow_constant_rewrite=not self.imports.shadowed_hf_cache_home,
            )
            if replacement is not None and isinstance(alias.name, cst.Name):
                changed = True
                self.auto_fixes += 1
                new_aliases.append(alias.with_changes(name=cst.Name(replacement)))
                code = "HF208" if imported == "hf_cache_home" else "HF101"
                rule = "constants-hf-cache-home" if imported == "hf_cache_home" else "rename-import"
                self.record(
                    original_alias,
                    code=code,
                    severity="info",
                    rule=rule,
                    message=f"Renamed {module_name}.{imported} import to {replacement}.",
                    auto_fixed=True,
                )
            else:
                new_aliases.append(alias)

        if not changed:
            return updated_node
        return updated_node.with_changes(names=tuple(new_aliases))

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        if (
            self.import_depth == 0
            and original_node.value == "hf_cache_home"
            and self.imports.hf_symbols.get("hf_cache_home") == "constants.hf_cache_home"
            and not self.imports.shadowed_hf_cache_home
            and not self.is_binding_name(original_node)
            and not self.is_shadowed("hf_cache_home")
        ):
            self.auto_fixes += 1
            self.record(
                original_node,
                code="HF208",
                severity="info",
                rule="constants-hf-cache-home",
                message="Replaced hf_cache_home with HF_HOME.",
                auto_fixed=True,
            )
            return cst.Name("HF_HOME")
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        raw_target = dotted_name(original_node.func)
        target = self.normalize_call_target(raw_target)
        if target is None:
            instance_get_token = hffolder_instance_get_token_target(original_node.func, self.imports)
            if instance_get_token is not None:
                return self.rewrite_hffolder_instance_get_token(original_node, updated_node, instance_get_token)
        if target is None or not target.startswith(f"{HF_MODULE}."):
            return updated_node

        if is_cached_download_target(target) and is_cached_download_hf_url_call(original_node, self.imports):
            return self.rewrite_cached_download_from_hf_url(original_node, updated_node, target)

        updated = updated_node
        updated = self.rewrite_call_target(original_node, updated, target)
        updated = self.rewrite_call_args(original_node, updated, target)
        self.flag_high_risk_calls(original_node, target)
        return updated

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.BaseExpression:
        target = normalize_target(dotted_name(original_node), self.imports)
        if target == f"{HF_MODULE}.constants.hf_cache_home":
            self.auto_fixes += 1
            self.record(
                original_node.attr,
                code="HF208",
                severity="info",
                rule="constants-hf-cache-home",
                message="Replaced constants.hf_cache_home with constants.HF_HOME.",
                auto_fixed=True,
            )
            return updated_node.with_changes(attr=cst.Name("HF_HOME"))
        return updated_node

    def leave_Try(self, original_node: cst.Try, updated_node: cst.Try) -> cst.Try:
        detector = HfCallDetector(self.imports, hfapi_names=self.current_hfapi_names())
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

    def normalize_call_target(self, raw_target: str | None) -> str | None:
        return normalize_hfapi_instance_target(
            normalize_target(raw_target, self.imports),
            raw_target,
            self.current_hfapi_names(),
        )

    def current_hfapi_names(self) -> set[str]:
        return self.hfapi_scopes[-1]

    def current_shadow_names(self) -> set[str]:
        return self.shadow_scopes[-1]

    def is_shadowed(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self.shadow_scopes))

    def is_binding_name(self, node: cst.Name) -> bool:
        try:
            parent = self.get_metadata(ParentNodeProvider, node)
        except KeyError:
            return False
        if isinstance(parent, (cst.AssignTarget, cst.Param, cst.FunctionDef, cst.ClassDef)):
            return True
        return isinstance(parent, cst.AnnAssign) and parent.target is node

    def track_hfapi_assignment(
        self, target: cst.BaseAssignTargetExpression, value: cst.BaseExpression
    ) -> None:
        if not isinstance(target, cst.Name):
            return

        target_name = target.value
        value_target = normalize_target(dotted_name(value.func), self.imports) if isinstance(value, cst.Call) else None
        if is_hfapi_constructor_target(value_target):
            self.current_hfapi_names().add(target_name)
        else:
            self.current_hfapi_names().discard(target_name)

    def track_shadow_assignment(self, target: cst.BaseAssignTargetExpression) -> None:
        if isinstance(target, cst.Name):
            self.current_shadow_names().add(target.value)

    def rewrite_hffolder_instance_get_token(
        self, original_node: cst.Call, updated_node: cst.Call, target: str
    ) -> cst.Call:
        if target == f"{HF_MODULE}.HfFolder.get_token":
            self.needed_hf_imports.add("get_token")
            new_func: cst.BaseExpression = cst.Name("get_token")
        else:
            module_alias = target[: -len(".HfFolder.get_token")].split(".")[-1]
            new_func = cst.Attribute(value=cst.Name(module_alias), attr=cst.Name("get_token"))

        self.auto_fixes += 1
        self.record(
            original_node.func,
            code="HF105",
            severity="info",
            rule="hffolder-instance-get-token",
            message="Replaced HfFolder().get_token() with get_token().",
            auto_fixed=True,
        )
        return updated_node.with_changes(func=new_func)

    def rewrite_cached_download_from_hf_url(
        self, original_node: cst.Call, updated_node: cst.Call, target: str
    ) -> cst.Call:
        if not is_cached_download_target(target):
            return updated_node
        if not is_cached_download_hf_url_call(original_node, self.imports):
            return updated_node

        self.needed_hf_imports.add("hf_hub_download")
        self.auto_fixes += 1
        self.record(
            original_node.func,
            code="HF106",
            severity="info",
            rule="cached-download-hf-url",
            message="Replaced cached_download(hf_hub_url(...)) with hf_hub_download(...).",
            auto_fixed=True,
        )

        updated_first_value = updated_node.args[0].value
        if not isinstance(updated_first_value, cst.Call):
            return updated_node

        outer_args: list[cst.Arg] = []
        for original_arg, arg in zip(original_node.args[1:], updated_node.args[1:]):
            keyword = original_arg.keyword.value if original_arg.keyword is not None else None
            if keyword == "use_auth_token":
                self.auto_fixes += 1
                outer_args.append(arg.with_changes(keyword=cst.Name("token")))
                self.record(
                    original_arg,
                    code="HF201",
                    severity="info",
                    rule="use-auth-token",
                    message="Renamed use_auth_token= to token=.",
                    auto_fixed=True,
                )
                continue
            if keyword in REMOVED_DOWNLOAD_KWS:
                self.auto_fixes += 1
                self.record(
                    original_arg,
                    code="HF202",
                    severity="info",
                    rule="removed-download-kw",
                    message=f"Removed deprecated {keyword}= argument from hf_hub_download().",
                    auto_fixed=True,
                )
                continue
            outer_args.append(arg)

        new_args = list(updated_first_value.args) + outer_args
        return updated_node.with_changes(
            func=cst.Name("hf_hub_download"),
            args=tuple(normalize_last_arg_comma(new_args, multiline=call_has_multiline_args(original_node))),
        )

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

        if target in {f"{HF_MODULE}.update_repo_visibility", f"{HF_MODULE}.HfApi.update_repo_visibility"}:
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
        safe_list_models_filter_keyword = safe_list_models_filter_arg(original_node, target)
        unresolved_cached_download = is_cached_download_target(target) and not is_cached_download_hf_url_call(
            original_node, self.imports
        )

        for original_arg, arg in zip(original_node.args, updated_node.args):
            if original_arg.star == "**" and is_dynamic_kwargs_review_target(target):
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

            if unresolved_cached_download:
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

            if is_list_models_target(target) and keyword in REMOVED_LIST_MODELS_FILTER_KWS:
                if keyword == safe_list_models_filter_keyword:
                    changed = True
                    self.auto_fixes += 1
                    new_args.append(arg.with_changes(keyword=cst.Name("filter")))
                    self.record(
                        original_arg,
                        code="HF205",
                        severity="info",
                        rule="list-models-single-filter",
                        message=f"Replaced list_models({keyword}=...) with list_models(filter=...).",
                        auto_fixed=True,
                    )
                    continue
                self.record(
                    original_arg,
                    code="HF402",
                    severity="warning",
                    rule="list-models-filter-review",
                    message=f"list_models({keyword}=...) changed in huggingface_hub v1.x; review combined filters manually instead of guessing.",
                    ai_recommended=True,
                )

            if is_build_hf_headers_target(target) and keyword == "is_write_action":
                changed = True
                self.auto_fixes += 1
                self.record(
                    original_arg,
                    code="HF206",
                    severity="info",
                    rule="build-hf-headers-is-write-action",
                    message="Removed deprecated build_hf_headers(is_write_action=...) argument.",
                    auto_fixed=True,
                )
                continue

            if target == f"{HF_MODULE}.AsyncInferenceClient" and keyword == "trust_env":
                if is_true(original_arg.value):
                    changed = True
                    self.auto_fixes += 1
                    self.record(
                        original_arg,
                        code="HF207",
                        severity="info",
                        rule="async-inference-client-trust-env",
                        message="Removed AsyncInferenceClient(trust_env=True); v1.x trusts environment variables by default.",
                        auto_fixed=True,
                    )
                    continue
                self.record(
                    original_arg,
                    code="HF404",
                    severity="warning",
                    rule="async-inference-client-trust-env-review",
                    message="AsyncInferenceClient(trust_env=...) changed in v1.x; review non-default environment trust behavior manually.",
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

        if target.endswith(".get_token_permission"):
            self.record(
                original_node,
                code="HF505",
                severity="warning",
                rule="get-token-permission-ai-review",
                message="get_token_permission was removed in v1.x; review permission-dependent token logic manually.",
                ai_recommended=True,
            )

        if is_cached_download_target(target) and not is_cached_download_hf_url_call(original_node, self.imports):
            self.record(
                original_node,
                code="HF504",
                severity="warning",
                rule="cached-download-ai-review",
                message="cached_download was removed; rewrite to hf_hub_download when the input is a Hub file, otherwise use an explicit HTTP/cache strategy.",
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


def hffolder_instance_get_token_target(func: cst.BaseExpression, imports: ImportState) -> str | None:
    if not isinstance(func, cst.Attribute) or func.attr.value != "get_token":
        return None
    if not isinstance(func.value, cst.Call):
        return None

    constructor = normalize_target(dotted_name(func.value.func), imports)
    if constructor == f"{HF_MODULE}.HfFolder":
        return f"{HF_MODULE}.HfFolder.get_token"
    if constructor is not None and constructor.endswith(".HfFolder"):
        return f"{constructor}.get_token"
    return None


def is_cached_download_hf_url_call(call: cst.Call, imports: ImportState) -> bool:
    if not call.args:
        return False
    first_arg = call.args[0]
    if first_arg.star or first_arg.keyword is not None or not isinstance(first_arg.value, cst.Call):
        return False
    inner_target = normalize_target(dotted_name(first_arg.value.func), imports)
    return is_hf_hub_url_target(inner_target)


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


class NameUsageCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.used: set[str] = set()
        self._import_depth = 0

    def visit_Import(self, node: cst.Import) -> bool | None:
        self._import_depth += 1
        return None

    def leave_Import(self, original_node: cst.Import) -> None:
        self._import_depth -= 1

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool | None:
        self._import_depth += 1
        return None

    def leave_ImportFrom(self, original_node: cst.ImportFrom) -> None:
        self._import_depth -= 1

    def visit_Name(self, node: cst.Name) -> None:
        if self._import_depth == 0:
            self.used.add(node.value)


class UnusedHfImportCleaner(cst.CSTTransformer):
    def __init__(self, used_names: set[str]) -> None:
        self.used_names = used_names

    def leave_SimpleStatementLine(
        self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
    ) -> cst.BaseStatement | cst.RemovalSentinel:
        if len(updated_node.body) != 1 or not isinstance(updated_node.body[0], cst.ImportFrom):
            return updated_node

        import_from = updated_node.body[0]
        module_name = dotted_name(import_from.module)
        if (
            module_name != HF_MODULE
            and not (module_name is not None and module_name.startswith(f"{HF_MODULE}."))
        ) or not isinstance(import_from.names, tuple):
            return updated_node

        aliases: list[cst.ImportAlias] = []
        for alias in import_from.names:
            imported = imported_symbol_name(alias)
            local = local_import_name(alias)
            if (
                imported in CLEANUP_HF_IMPORTS_IF_UNUSED
                and local is not None
                and local not in self.used_names
            ):
                continue
            aliases.append(alias)

        if not aliases:
            return cst.RemoveFromParent()
        return updated_node.with_changes(body=[import_from.with_changes(names=tuple(aliases))])


def cleanup_unused_hf_imports(code: str) -> str:
    module = cst.parse_module(code)
    collector = NameUsageCollector()
    module.visit(collector)
    cleaned = module.visit(UnusedHfImportCleaner(collector.used))
    return cleaned.code


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
    code = cleanup_unused_hf_imports(updated.code)

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
