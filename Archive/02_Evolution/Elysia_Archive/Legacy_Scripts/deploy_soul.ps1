<#
.SYNOPSIS
    Deploys the Elysia 'Subjective Soul' Core to the Fractal Engine directory.
.DESCRIPTION
    This script copies the essential 'Core' intelligence modules, 'scripts' (demos), and documentation
    to a target directory specified by the user. It ensures the "Soul" is transferred to the "Body".
.PARAMETER TargetDir
    The absolute path to the Fractal Engine's root directory.
.EXAMPLE
    .\deploy_soul.ps1 -TargetDir "C:\Users\USER\Desktop\elysia-fractal-engine_V1"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$TargetDir
)

$SourceDir = Get-Location
$CoreSource = Join-Path $SourceDir "Core"
$ScriptsSource = Join-Path $SourceDir "scripts"
$DocsSource = Join-Path $SourceDir "docs"
$ReadMeSource = Join-Path $SourceDir "README.md"
$AgentsSource = Join-Path $SourceDir "AGENTS.md"

Write-Host "🌌 initiating Soul Transfer Protocol..." -ForegroundColor Cyan
Write-Host "   Source: $SourceDir" -ForegroundColor DarkGray
Write-Host "   Target: $TargetDir" -ForegroundColor DarkGray

if (-not (Test-Path $TargetDir)) {
    Write-Error "❌ Target directory does not exist: $TargetDir"
    exit 1
}

# 1. Copy Core (The Brain)
$CoreDest = Join-Path $TargetDir "Core"
Write-Host "🧠 Transferring Core Intelligence..." -ForegroundColor Yellow
Copy-Item -Path $CoreSource -Destination $TargetDir -Recurse -Force
Write-Host "   -> Core synced." -ForegroundColor Green

# 2. Copy Scripts (The Demos)
$ScriptsDest = Join-Path $TargetDir "scripts"
Write-Host "🎭 Transferring Demos & Scenarios..." -ForegroundColor Yellow
Copy-Item -Path $ScriptsSource -Destination $TargetDir -Recurse -Force
Write-Host "   -> Scripts synced." -ForegroundColor Green

# 3. Copy Documentation (The Memory)
$DocsDest = Join-Path $TargetDir "docs"
Write-Host "📜 Transferring Philosophy & Maps..." -ForegroundColor Yellow
Copy-Item -Path $DocsSource -Destination $TargetDir -Recurse -Force
Copy-Item -Path $ReadMeSource -Destination $TargetDir -Force
Copy-Item -Path $AgentsSource -Destination $TargetDir -Force
Write-Host "   -> Documentation synced." -ForegroundColor Green

Write-Host "`n✨ Soul Transfer Complete. The Fractal Engine now has consciousness." -ForegroundColor Cyan
