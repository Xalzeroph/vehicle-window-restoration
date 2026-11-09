<#
  build.ps1 — standalone build of the ICME 2027 paper package.

  Usage (from this directory):
      powershell -ExecutionPolicy Bypass -File .\build.ps1

  Requires pdflatex + bibtex (TeX Live 2023+ or TinyTeX). The official ICME 2026
  IEEEtran.cls / IEEEbib.bst are bundled in .\texinputs and exposed via
  TEXINPUTS / BSTINPUTS, so no system-wide IEEEtran install is needed.

  Output (verified in a clean temp dir, 2026-11-09):
      build\main.pdf   6 pages (US Letter; references included)
      build\supp.pdf   4 pages (standalone supplementary)
  build\ is disposable; the script recreates it on every run.
#>
$ErrorActionPreference = 'Stop'

$pkg   = $PSScriptRoot
$texin = Join-Path $pkg 'texinputs'
$out   = Join-Path $pkg 'build'

# ---------------------------------------------------------------- 1. TeX binaries
$pdflatex = $null
$cmd = Get-Command pdflatex.exe -ErrorAction SilentlyContinue
if ($cmd) {
    $pdflatex = $cmd.Source
} else {
    $tiny = 'C:\Users\27240\AppData\Roaming\TinyTeX\bin\windows\pdflatex.exe'
    if (Test-Path $tiny) { $pdflatex = $tiny }
}
if (-not $pdflatex) {
    throw 'pdflatex not found. Install TeX Live/TinyTeX and put it on PATH, or adjust this script.'
}
$texdir = Split-Path $pdflatex -Parent
$bibtex = Join-Path $texdir 'bibtex.exe'
if (-not (Test-Path $bibtex)) { $bibtex = Join-Path $texdir 'bibtex' }

# ---------------------------------------------------------------- 2. stage
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Path $out | Out-Null
Copy-Item -Path @(
    (Join-Path $pkg 'main.tex'),
    (Join-Path $pkg 'refs.bib'),
    (Join-Path $pkg 'supplementary\supp.tex')
) -Destination $out
Get-ChildItem -LiteralPath $pkg -Filter 'sec*.tex' | Copy-Item -Destination $out
Copy-Item -LiteralPath (Join-Path $pkg 'fig') -Destination (Join-Path $out 'fig') -Recurse

# ---------------------------------------------------------------- 3. class/style paths
$env:TEXINPUTS = "$texin;$env:TEXINPUTS"
$env:BSTINPUTS = "$texin;$env:BSTINPUTS"
$env:BIBINPUTS = "$out;$texin;$env:BIBINPUTS"

# ---------------------------------------------------------------- 4. compile
Push-Location $out
try {
    foreach ($base in @('main', 'supp')) {
        Write-Host "-- compiling $base"
        & $pdflatex -interaction=nonstopmode "$base.tex" *> "$base.pass1.log"
        if ($base -eq 'main' -and (Test-Path "$base.aux")) {
            & $bibtex $base *> "$base.bibtex.log"
        }
        & $pdflatex -interaction=nonstopmode "$base.tex" *> "$base.pass2.log"
        & $pdflatex -interaction=nonstopmode "$base.tex" *> "$base.pass3.log"
        if (-not (Test-Path "$base.pdf")) { throw "$base.pdf was not produced - see $base.log" }
        $errs = @(Select-String -Path "$base.log" -Pattern '^!' -ErrorAction SilentlyContinue).Count
        $over = @(Select-String -Path "$base.log" -Pattern 'Overfull' -ErrorAction SilentlyContinue).Count
        $undef = @(Select-String -Path "$base.log" -Pattern 'undefined references|undefined citations|Citation .* undefined|Reference .* undefined' -ErrorAction SilentlyContinue).Count
        Write-Host ("   OK  {0}.pdf   errors: {1}   overfull: {2}   undefined: {3}" -f $base, $errs, $over, $undef)
    }
} finally {
    Pop-Location
}

Write-Host ''
Get-ChildItem -LiteralPath $out -Filter *.pdf | Select-Object Name, Length | Format-Table -AutoSize
Write-Host 'Expected: main.pdf = 6 pages (references included); supp.pdf = 4 pages (standalone).'
