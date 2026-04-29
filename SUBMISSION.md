# Boring AI Submission Notes

## Project

Hugging Face Hub v1 Migrator

## Goal

Migrate Python projects from `huggingface_hub` v0.x call sites to v1.x-compatible APIs with a conservative AI-assisted codemod:

- deterministic LibCST rewrites for high-confidence API changes
- AI suggestions only for semantic migrations that are unsafe to rewrite blindly
- structured JSON report for auditability

## Why this fits Boring AI

This is not a chatbot wrapper. The codemod uses concrete syntax tree transformations to preserve formatting and comments, then limits AI to review-only suggestions for low-confidence blocks.

## Automatic fixes

- `use_auth_token=` -> `token=`
- `hf_hub_download(..., resume_download=...)` -> removed argument
- `hf_hub_download(..., force_filename=...)` -> removed argument
- `hf_hub_download(..., local_dir_use_symlinks=...)` -> removed argument
- `InferenceApi(...)` -> `InferenceClient(...)`
- `update_repo_visibility(...)` -> `update_repo_settings(...)`
- `HfFolder.get_token()` -> `get_token()`
- `except requests.HTTPError` around Hugging Face Hub calls -> `except HfHubHttpError`
- `login(write_permission=...)` -> removed argument
- `login(new_session=True/False)` -> `skip_if_logged_in=False/True`

## AI-assisted review

AI is triggered for high-risk findings and writes suggestions to the JSON report. It does not overwrite source files.

Current review-only cases:

- `Repository(...)`, because migration depends on upload/download intent
- dynamic `**kwargs` passed to known high-risk Hugging Face Hub APIs
- changed `list_models(...)` filters
- `configure_http_backend(...)`
- non-trivial `HfFolder` token flows

## Validation

Local fixture suite:

```text
8 passed
```

AI fallback smoke test:

```text
1 Repository(...) finding
ai_fallback.configured = true
ai_suggestion generated = true
```

Real project dry-run:

```text
Target: datasets==2.14.0 source distribution from PyPI
156 Python files scanned
2 files changed
2 deterministic fixes
0 AI-review noise after restricting dynamic **kwargs checks
156 transformed modules compile successfully in memory
```

Representative real-project diff:

```diff
-from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HfFolder
+from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HfFolder, get_token

-token = token if token is not None else HfFolder.get_token()
+token = token if token is not None else get_token()
```

## Run

```bash
pip install -e ".[test]"
pytest
hf-hub-v1-migrator path/to/repo --diff --report hf-v1-report.json
hf-hub-v1-migrator path/to/repo --write --report hf-v1-report.json
hf-hub-v1-migrator path/to/repo --report hf-v1-report.json --ai-fallback
```

## Safety boundaries

- The AI path is review-only.
- Low-confidence code is reported instead of rewritten.
- Deterministic output is idempotent in tests.
- Golden outputs are compiled with `compile(..., mode="exec")`.
- Secrets are read from `.env` or environment variables and are excluded from git.

