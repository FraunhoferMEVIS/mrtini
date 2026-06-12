$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
$External = Join-Path $Root "external"

$Repo = "https://github.com/FraunhoferMEVIS/gammaSTAR.git"
$GammaStarRef = if ($env:GAMMASTAR_REF) { $env:GAMMASTAR_REF } else { "main" }

$Source = Join-Path $External ".gammaSTAR_source"
$Subdir = "sequenceServer/gammastar_recon"
$Dest = Join-Path $External "gammastar_recon"

New-Item -ItemType Directory -Force -Path $External | Out-Null

if (-not (Test-Path (Join-Path $Source ".git"))) {
    git clone --filter=blob:none --sparse $Repo $Source
}

git -C $Source fetch origin $GammaStarRef
git -C $Source checkout FETCH_HEAD
git -C $Source sparse-checkout set $Subdir

if (Test-Path $Dest) {
    Remove-Item -Recurse -Force $Dest
}
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

Copy-Item -Path (Join-Path $Source "$Subdir\*") -Destination $Dest -Recurse -Force

Write-Host "gammaSTAR recon ref:"
git -C $Source rev-parse --short HEAD
Write-Host "gammaSTAR recon available at:"
Write-Host $Dest