param(
    [string]$Python = "python",
    [switch]$AiFallback,
    [switch]$ApplyAi,
    [string]$EnvFile = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$demoSource = Join-Path $root "examples\demo_repo"
$demoWork = Join-Path $root "tmp-demo-repo"
$report = Join-Path $root "hf-v1-demo-report.json"
$aiReport = Join-Path $root "hf-v1-ai-demo-report.json"
$deterministicFile = Join-Path $demoWork "model_download.py"
$aiFile = Join-Path $demoWork "push_model.py"

if (-not $EnvFile) {
    $rootEnv = Join-Path $root ".env"
    $legacyEnv = Join-Path $root "hf-hub-v1-migrator\.env"
    if (Test-Path -LiteralPath $rootEnv) {
        $EnvFile = $rootEnv
    } elseif (Test-Path -LiteralPath $legacyEnv) {
        $EnvFile = $legacyEnv
    } else {
        $EnvFile = $rootEnv
    }
}

if (Test-Path -LiteralPath $demoWork) {
    Remove-Item -LiteralPath $demoWork -Recurse -Force
}
if (Test-Path -LiteralPath $report) {
    Remove-Item -LiteralPath $report -Force
}
if (Test-Path -LiteralPath $aiReport) {
    Remove-Item -LiteralPath $aiReport -Force
}

Write-Host ""
Write-Host "0) Install local package for this Python"
Write-Host "------------------------------------------------------------"
& $Python -m pip install -q -e $root
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install hf-hub-v1-migrator into the selected Python. Try passing -Python with a working Python executable."
}

Copy-Item -LiteralPath $demoSource -Destination $demoWork -Recurse

Write-Host ""
Write-Host "============================================================"
Write-Host "HF Hub v1 Migrator Demo"
Write-Host "============================================================"
Write-Host ""

Write-Host "1) Legacy code before migration"
Write-Host "------------------------------------------------------------"
Get-Content -LiteralPath (Join-Path $demoWork "model_download.py")

Write-Host ""
Write-Host "2) Deterministic dry-run diff"
Write-Host "------------------------------------------------------------"
& $Python -m hf_hub_v1_migrator $deterministicFile --diff --report $report
if ($LASTEXITCODE -ne 0) { throw "Dry-run migration failed." }

Write-Host ""
Write-Host "3) Apply deterministic migration"
Write-Host "------------------------------------------------------------"
& $Python -m hf_hub_v1_migrator $deterministicFile --write --report $report
if ($LASTEXITCODE -ne 0) { throw "Write migration failed." }

Write-Host ""
Write-Host "4) Migrated code after --write"
Write-Host "------------------------------------------------------------"
Get-Content -LiteralPath (Join-Path $demoWork "model_download.py")

Write-Host ""
Write-Host "5) Structured report summary"
Write-Host "------------------------------------------------------------"
& $Python -c "import json; data=json.load(open(r'$report', encoding='utf-8')); s=data['summary']; print('files_scanned =', s['files_scanned']); print('files_changed =', s['files_changed']); print('deterministic_fixes =', s['auto_fixes']); print('syntax_safe = checked in next step')"
if ($LASTEXITCODE -ne 0) { throw "Report summary failed." }

Write-Host ""
Write-Host "6) Compile migrated demo repo"
Write-Host "------------------------------------------------------------"
& $Python -m compileall -q $demoWork
if ($LASTEXITCODE -ne 0) { throw "compileall failed." }
Write-Host "compileall passed"

if ($AiFallback) {
    Write-Host ""
    Write-Host "7) AI fallback for Repository(...) review case"
    Write-Host "------------------------------------------------------------"
    $aiArgs = @("-m", "hf_hub_v1_migrator", $aiFile, "--report", $aiReport, "--ai-fallback", "--env-file", $EnvFile)
    if ($ApplyAi) {
        $aiArgs += "--apply-ai"
    }
    & $Python @aiArgs
    if ($LASTEXITCODE -ne 0) { throw "AI fallback run failed." }
    & $Python -c "import json; data=json.load(open(r'$aiReport', encoding='utf-8')); f=data['files'][0]['findings'][0]; print('env_file =', r'$EnvFile'); print('ai_configured =', data['ai_fallback']['configured']); print('apply_ai =', data['ai_fallback'].get('apply_requested')); print('finding =', f['code'], f['rule']); print('ai_suggestion_generated =', bool(f.get('ai_suggestion'))); print('ai_proposal_action =', (f.get('ai_proposal') or {}).get('action')); print('ai_review_decision =', (f.get('ai_review') or {}).get('decision')); print('ai_review_risk =', (f.get('ai_review') or {}).get('risk')); print('ai_applied =', f.get('ai_applied')); print('ai_apply_reason =', f.get('ai_apply_reason')); print('ai_suggestion_preview ='); print((f.get('ai_suggestion') or '')[:500])"
    if ($LASTEXITCODE -ne 0) { throw "AI report summary failed." }
}

Write-Host ""
Write-Host "Demo files:"
Write-Host $demoWork
Write-Host $report
if ($AiFallback) {
    Write-Host $aiReport
}
