# Publishing

This package is structured for the Codemod Registry:

- `codemod.yaml` contains package metadata.
- `workflow.yaml` contains the registry workflow.
- `README.md`, `SUBMISSION.md`, and `LICENSE` are included for review and discoverability.

## Local publish

Use this when publishing manually from your machine:

```bash
npx codemod login
npx codemod whoami
npx codemod publish
```

## API key publish

Use this in CI or any non-interactive environment:

```bash
CODEMOD_AUTH_TOKEN=$CODEMOD_API_KEY npx codemod publish
```

Do not commit the API key. Store it as a CI secret.

## Trusted publisher

If this is moved to a GitHub organization whose name matches the package scope, use OIDC trusted publishing:

```yaml
name: Publish Codemod
on:
  push:
    tags:
      - "v*"

permissions:
  id-token: write
  contents: read

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Publish codemod
        uses: codemod/publish-action@v1
```

## Before publishing

Replace the placeholder `author` in `codemod.yaml` with your real name or organization.

Run:

```bash
python -m pip install -e ".[test]"
pytest
python -m hf_hub_v1_migrator tests/fixtures --diff --report hf-v1-demo-report.json
```

Do not publish:

- `.env`
- `hf-v1-*.json`
- `pytest-cache-files-*`
- `real-repos/`
