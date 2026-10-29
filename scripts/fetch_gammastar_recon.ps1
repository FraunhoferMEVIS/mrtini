$ErrorActionPreference = "Stop"
& python (Join-Path $PSScriptRoot "fetch_gammastar_recon.py") @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
