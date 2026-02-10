# Script PowerShell untuk membuka Emotion Classification App
# Jalankan dengan: .\open-app.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Opening Emotion Classification App" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path

# Open frontend in default browser
$frontendPath = Join-Path $scriptPath "frontend\index.html"

if (Test-Path $frontendPath) {
    Write-Host "✅ Opening frontend application..." -ForegroundColor Green
    Start-Process $frontendPath
    Write-Host ""
    Write-Host "✅ Application opened in your default browser!" -ForegroundColor Green
    Write-Host "📡 Backend API: http://localhost:5000" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "❌ Error: frontend/index.html not found!" -ForegroundColor Red
    Write-Host "   Current path: $scriptPath" -ForegroundColor Yellow
}

Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
