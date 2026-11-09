# Build the CVPR 2027 submission with the local TinyTeX installation.
$ErrorActionPreference = "Stop"
$tex = "C:\Users\27240\AppData\Roaming\TinyTeX\bin\windows"
if (-not (Test-Path "$tex\pdflatex.exe")) {
    Write-Error "TinyTeX not found at $tex. Install it from https://yihui.org/tinytex/ or use Overleaf."
}
Set-Location $PSScriptRoot
foreach ($pass in 1, 2) {
    & "$tex\pdflatex.exe" -interaction=nonstopmode main.tex | Out-Null
}
& "$tex\bibtex.exe" main | Out-Null
foreach ($pass in 1, 2) {
    & "$tex\pdflatex.exe" -interaction=nonstopmode main.tex | Out-Null
}
$log = Get-Content "main.log" -Raw
$m = [regex]::Match($log, "Output written on main\.pdf \((\d+) pages")
if ($m.Success) {
    Write-Host "PDF: main.pdf  ($($m.Groups[1].Value) pages)" -ForegroundColor Green
} else {
    Write-Host "Build failed - see main.log" -ForegroundColor Red
    Get-Content "main.log" | Select-String -Pattern "^!" | Select-Object -First 5
}
$over = (Get-Content "main.log" -Raw).Split("`n") | Where-Object { $_ -match "Overfull \\hbox|Float too large" }
if ($over) {
    Write-Host "Layout warnings:" -ForegroundColor Yellow
    $over | Select-Object -First 5 | ForEach-Object { Write-Host "  $_" }
}
