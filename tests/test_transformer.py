from __future__ import annotations

from pathlib import Path

from hf_hub_v1_migrator.ai_reviewer import maybe_apply_proposal
from hf_hub_v1_migrator.report import Finding
from hf_hub_v1_migrator import transform_source


FIXTURES = Path(__file__).parent / "fixtures"


def read_fixture(name: str, file_name: str) -> str:
    return (FIXTURES / name / file_name).read_text(encoding="utf-8")


def assert_golden(name: str) -> None:
    before = read_fixture(name, "before.py")
    expected = read_fixture(name, "after.py")

    result = transform_source(before, path=f"{name}/before.py")
    assert result.code == expected
    compile(result.code, filename=f"{name}/after.py", mode="exec")

    second = transform_source(result.code, path=f"{name}/after.py")
    assert second.code == result.code


def test_use_auth_token_and_removed_download_kwargs() -> None:
    assert_golden("use_auth_token_to_token")


def test_inference_api_is_renamed() -> None:
    assert_golden("inference_api")


def test_hffolder_get_token_is_rewritten() -> None:
    assert_golden("hffolder_get_token")


def test_hffolder_instance_get_token_is_rewritten() -> None:
    assert_golden("hffolder_instance_get_token")


def test_cached_download_hf_url_is_rewritten() -> None:
    assert_golden("cached_download_hf_url")


def test_cached_download_hf_url_submodule_import_is_rewritten() -> None:
    assert_golden("cached_download_submodule_hf_url")


def test_requests_exceptions_http_error_is_rewritten_in_hf_try_block() -> None:
    assert_golden("requests_exceptions_http_error")


def test_hfapi_instance_methods_are_rewritten_after_local_constructor() -> None:
    assert_golden("hfapi_instance_methods")


def test_list_models_single_removed_filter_is_rewritten() -> None:
    assert_golden("list_models_single_filter")


def test_build_hf_headers_is_write_action_is_removed() -> None:
    assert_golden("build_hf_headers_is_write_action")


def test_async_inference_client_trust_env_true_is_removed() -> None:
    assert_golden("async_inference_trust_env")


def test_constants_hf_cache_home_is_rewritten() -> None:
    assert_golden("constants_hf_cache_home")


def test_constants_hf_cache_home_direct_import_is_rewritten() -> None:
    assert_golden("constants_hf_cache_home_direct_import")


def test_requests_http_error_is_only_rewritten_in_hf_try_block() -> None:
    assert_golden("http_error_exception")


def test_repository_is_reported_not_rewritten() -> None:
    before = read_fixture("repository_report_only", "before.py")
    result = transform_source(before, path="repository_report_only/before.py")

    assert result.code == before
    assert any(finding.code == "HF501" and finding.ai_recommended for finding in result.findings)


def test_dynamic_kwargs_are_reported_not_rewritten() -> None:
    before = read_fixture("dynamic_kwargs_report_only", "before.py")
    result = transform_source(before, path="dynamic_kwargs_report_only/before.py")

    assert result.code == before
    assert any(finding.code == "HF401" and finding.ai_recommended for finding in result.findings)


def test_cached_download_url_is_reported_not_rewritten() -> None:
    before = read_fixture("cached_download_report_only", "before.py")
    result = transform_source(before, path="cached_download_report_only/before.py")

    assert result.code == before
    assert any(finding.code == "HF504" and finding.ai_recommended for finding in result.findings)


def test_cached_download_url_kwargs_are_reported_not_partially_rewritten() -> None:
    before = read_fixture("cached_download_kwargs_report_only", "before.py")
    result = transform_source(before, path="cached_download_kwargs_report_only/before.py")

    assert result.code == before
    assert result.auto_fixes == 0
    assert any(finding.code == "HF504" and finding.ai_recommended for finding in result.findings)


def test_list_models_combined_removed_filters_are_reported_not_rewritten() -> None:
    before = read_fixture("list_models_combined_review_only", "before.py")
    result = transform_source(before, path="list_models_combined_review_only/before.py")

    assert result.code == before
    assert sum(1 for finding in result.findings if finding.code == "HF402") == 2


def test_get_token_permission_is_reported_not_rewritten() -> None:
    before = read_fixture("get_token_permission_report_only", "before.py")
    result = transform_source(before, path="get_token_permission_report_only/before.py")

    assert result.code == before
    assert any(finding.code == "HF505" and finding.ai_recommended for finding in result.findings)


def test_negative_same_keyword_non_hf_call_is_unchanged() -> None:
    before = read_fixture("negative_non_hf_keyword", "before.py")
    result = transform_source(before, path="negative_non_hf_keyword/before.py")

    assert result.code == before
    assert result.auto_fixes == 0
    assert result.findings == []


def test_negative_requests_http_error_without_hf_call_is_unchanged() -> None:
    before = read_fixture("negative_requests_exception", "before.py")
    result = transform_source(before, path="negative_requests_exception/before.py")

    assert result.code == before
    assert result.auto_fixes == 0


def test_negative_reassigned_hfapi_variable_is_unchanged() -> None:
    before = read_fixture("negative_hfapi_reassigned", "before.py")
    result = transform_source(before, path="negative_hfapi_reassigned/before.py")

    assert result.code == before
    assert result.auto_fixes == 0


def test_negative_hf_cache_home_shadow_is_not_rewritten() -> None:
    before = read_fixture("negative_hf_cache_home_shadow", "before.py")
    result = transform_source(before, path="negative_hf_cache_home_shadow/before.py")

    assert result.code == before
    assert result.auto_fixes == 0


def test_ai_patch_applies_only_bounded_low_risk_function() -> None:
    source = """from huggingface_hub import Repository


def push_model(local_dir: str):
    repo = Repository(local_dir=local_dir, clone_from="user/model")
    repo.git_add()
    repo.git_commit("update")
    repo.git_push()


def unrelated():
    return "keep me"
"""
    finding = Finding(
        code="HF501",
        severity="warning",
        path="example.py",
        line=5,
        column=11,
        message="Repository needs migration",
        rule="repository-ai-review",
        ai_recommended=True,
    )
    proposal = {
        "action": "replace_block",
        "target": {"type": "function", "line": 5},
        "replacement": """def push_model(local_dir: str):
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(folder_path=local_dir, repo_id="user/model")""",
        "summary": "Use HfApi.upload_folder.",
        "assumptions": ["local_dir is the folder to upload"],
        "risk_factors": [],
    }
    review = {
        "decision": "apply",
        "risk": 0.1,
        "reason": "local function rewrite",
        "required_human_checks": [],
    }

    code, applied, reason = maybe_apply_proposal(source, finding, proposal, review, risk_threshold=0.35)

    assert applied, reason
    assert "api.upload_folder" in code
    assert "def unrelated()" in code
    assert "keep me" in code


def test_ai_patch_rejects_high_risk_review() -> None:
    source = """from huggingface_hub import Repository


def push_model(local_dir: str):
    repo = Repository(local_dir=local_dir, clone_from="user/model")
    repo.git_push()
"""
    finding = Finding(
        code="HF501",
        severity="warning",
        path="example.py",
        line=5,
        column=11,
        message="Repository needs migration",
        rule="repository-ai-review",
        ai_recommended=True,
    )
    proposal = {
        "action": "replace_block",
        "target": {"type": "function", "line": 5},
        "replacement": "def push_model(local_dir: str):\n    pass",
    }
    review = {"decision": "manual_review", "risk": 0.8, "reason": "ambiguous", "required_human_checks": ["repo id"]}

    code, applied, reason = maybe_apply_proposal(source, finding, proposal, review, risk_threshold=0.35)

    assert not applied
    assert "manual review" in reason
    assert code == source
