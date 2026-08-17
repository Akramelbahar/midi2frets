# Convenience runner for Windows PowerShell.
#
#   .\train.ps1 overfit     # quick 60s sanity check (expect >95% acc on file.json)
#   .\train.ps1 train       # train on the GP corpus with live insight logging
#   .\train.ps1 preprocess  # finish GP -> JSON conversion (resumable)
#   .\train.ps1 all         # preprocess remaining files, then train
#
# Any extra flags are passed through to run.py, e.g.:
#   .\train.ps1 train --max-files 500 --epochs 5
#   .\train.ps1 train --resume checkpoints/model_gp.pt.resume
#
# Tail the live log in a second terminal:
#   Get-Content checkpoints\logs\training.log -Wait -Tail 40

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("preprocess", "manifest", "overfit", "train", "all")]
    [string]$Stage,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "=== midi2Frets :: stage '$Stage' ===" -ForegroundColor Cyan
python run.py $Stage @Rest
