from __future__ import annotations

from pathlib import Path

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
