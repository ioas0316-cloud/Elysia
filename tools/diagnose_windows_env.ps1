param(
  [string]$Python = ".venv\Scripts\python.exe",
  [string[]]$Drivers = @('windows','windib','directx'),
  [int]$Size = 800,
  [int]$Seconds = 4
)

# UTF-8 출력 강제 (VSCode/PowerShell 한글 깨짐 방지)
try { chcp.com 65001 > $null 2>&1 } catch {}
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONIOENCODING = 'utf-8'

Write-Host "[진단] Pygame/SDL 윈도우 드라이버 환경 점검 시작" -ForegroundColor Cyan
Write-Host "        Python  : $Python"
Write-Host "        Drivers : $($Drivers -join ', ')"
Write-Host "        Size    : $Size"
Write-Host "        Seconds : $Seconds"

$script = "ElysiaStarter\scripts\diagnose_pygame_env.py"
if (!(Test-Path $script)) {
  Write-Host "[오류] 테스트 스크립트를 찾을 수 없습니다: $script" -ForegroundColor Red
  exit 2
}

$results = @()
foreach ($drv in $Drivers) {
  Write-Host "`n=== 드라이버 시도: $drv ===" -ForegroundColor Yellow
  $env:SDL_VIDEODRIVER = $drv
  $env:SDL_AUDIODRIVER = 'dummy'
  try {
    & $Python $script --size $Size --seconds $Seconds
    $code = $LASTEXITCODE
  } catch {
    $code = 1
  }
  if ($code -eq 0) {
    Write-Host "[성공] $drv 드라이버에서 창 생성/유지 OK" -ForegroundColor Green
  } else {
    Write-Host "[실패] $drv 드라이버에서 창 생성 실패 또는 즉시 종료" -ForegroundColor Red
  }
  $results += [pscustomobject]@{ Driver=$drv; ExitCode=$code }
}

Write-Host "`n요약:" -ForegroundColor Cyan
$results | ForEach-Object {
  $msg = if ($_.ExitCode -eq 0) { 'OK' } else { 'FAIL' }
  Write-Host (" - {0,-8} : {1}" -f $_.Driver, $msg)
}

if ($results.Where({ $_.ExitCode -eq 0 }).Count -gt 0) {
  Write-Host "`n[안내] 위에서 성공한 드라이버를 SDL_VIDEODRIVER 기본값으로 사용할 것을 권장합니다." -ForegroundColor Cyan
  Write-Host "      예) PowerShell에서:  `\$env:SDL_VIDEODRIVER='windows'; .\start.bat"
  exit 0
} else {
  Write-Host "`n[점검 필요] 모든 드라이버에서 실패했습니다. GPU/원격/가상환경 설정 확인이 필요합니다." -ForegroundColor Red
  Write-Host "      logs\starter_debug.log 과 logs\diag_pygame_env.log 파일을 함께 공유해 주세요."
  exit 1
}

