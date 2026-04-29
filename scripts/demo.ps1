param(
    [string]$Python = "python",
    [string]$Target = ".\tests\fixtures",
    [switch]$AiFallback
)

$ErrorActionPreference = "Stop"

Write-Host "== Install editable package =="
& $Python -m pip install -e ".[test]"

Write-Host "== Run fixture suite =="
$env:PYTHONDONTWRITEBYTECODE = "1"
& $Python -m pytest .\tests -q -p no:cacheprovider

Write-Host "== Run codemod dry-run =="
& $Python -m hf_hub_v1_migrator $Target --diff --report .\hf-v1-demo-report.json

if ($AiFallback) {
    Write-Host "== Run AI fallback smoke test =="
    & $Python -m hf_hub_v1_migrator .\tests\fixtures\repository_report_only\before.py --report .\hf-v1-ai-demo-report.json --ai-fallback --env-file .\.env
}

Write-Host "Demo complete."
