# Boring AI Submission Notes

## Project

Hugging Face Hub v1 Migrator

## Goal

Migrate Python projects from `huggingface_hub` v0.x call sites to v1.x-compatible APIs with a conservative AI-assisted codemod:

- deterministic LibCST rewrites for high-confidence API changes
- AI suggestions only for semantic migrations that are unsafe to rewrite blindly
- optional AI JSON patch proposals with a second AI risk review before any bounded local application
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
- selected locally constructed `HfApi()` instance methods, including `use_auth_token=` and `update_repo_visibility(...)`
- `HfFolder.get_token()` -> `get_token()`
- `except requests.HTTPError` around Hugging Face Hub calls -> `except HfHubHttpError`
- `login(write_permission=...)` -> removed argument
- `login(new_session=True/False)` -> `skip_if_logged_in=False/True`
- single `list_models(task/library/language/tags=...)` filter -> `filter=...`
- `build_hf_headers(is_write_action=...)` -> removed argument
- `AsyncInferenceClient(trust_env=True)` -> removed argument
- `constants.hf_cache_home` -> `constants.HF_HOME`

## AI-assisted review

AI is triggered for high-risk findings and writes suggestions to the JSON report. With `--apply-ai`, the first model must return a strict JSON proposal and a second AI reviewer must approve the patch with a low risk score. Even then, the tool only replaces one local function or statement block and verifies the result with LibCST and Python `compile()`.

Current review-only cases:

- `Repository(...)`, because migration depends on upload/download intent
- dynamic `**kwargs` passed to known high-risk Hugging Face Hub APIs
- combined `list_models(...)` filters
- `configure_http_backend(...)`
- non-trivial `HfFolder` token flows
- dynamic or false `AsyncInferenceClient(trust_env=...)`
- removed `get_token_permission(...)` permission logic

## Validation

Local fixture suite:

```text
26 passed
```

AI fallback smoke test:

```text
1 Repository(...) finding
ai_fallback.configured = true
ai_suggestion generated = true
structured proposal/review generated = true
high-risk Repository(...) proposal held for manual review when reviewer risk was high
```

Real package benchmark:

```text
Targets: sentence-transformers==2.2.2, peft==0.4.0, diffusers==0.16.1 source distributions
315 Python files scanned
11 files changed
30 deterministic fixes
8 AI/manual review findings
315 transformed modules compile successfully in memory
0 syntax failures
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
- `--apply-ai` is opt-in and bounded to local block replacement.
- AI output must pass JSON schema, risk review, LibCST parsing, and Python compilation.
- Low-confidence code is reported instead of rewritten.
- Deterministic output is idempotent in tests.
- Golden outputs are compiled with `compile(..., mode="exec")`.
- `scripts/benchmark_coverage.py` compiles every transformed module in real package dry-runs.
- Secrets are read from `.env` or environment variables and are excluded from git.

