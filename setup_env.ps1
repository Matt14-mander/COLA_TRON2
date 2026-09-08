# COLA/TRON2 environment setup for PowerShell. Dot-source this file:
#   . .\setup_env.ps1

$repoRoot = $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'rsl_rl\setup.py'))) {
    throw 'Missing rsl_rl submodule; run: git submodule update --init --recursive'
}

$env:COLA_ROOT = $repoRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot;$env:PYTHONPATH" } else { $repoRoot }
$env:COLA_ISAACLAB_LOG_DIR = Join-Path $repoRoot 'logs\isaaclab'
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
if (-not $env:WANDB_MODE) { $env:WANDB_MODE = 'offline' }

# Warp's Windows native compiler cannot reliably create PTX files below a
# path containing non-ASCII characters. Prefer the caller's override; use the
# first writable ASCII location below the repository otherwise.
if (-not $env:WARP_CACHE_PATH) {
    $asciiCache = Join-Path $env:LOCALAPPDATA 'COLA_TRON2\warp-cache'
    if ($asciiCache -notmatch '[^\x00-\x7F]') {
        $env:WARP_CACHE_PATH = $asciiCache
    } else {
        throw 'Set WARP_CACHE_PATH to a writable ASCII-only directory before running Isaac Sim.'
    }
}
New-Item -ItemType Directory -Force -Path $env:WARP_CACHE_PATH | Out-Null

Write-Host "[setup] COLA_ROOT=$env:COLA_ROOT"
Write-Host "[setup] WARP_CACHE_PATH=$env:WARP_CACHE_PATH"
