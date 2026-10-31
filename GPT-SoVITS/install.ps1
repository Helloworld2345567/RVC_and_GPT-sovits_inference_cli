[CmdletBinding()]
param(
    [ValidateSet("CU129", "CPU")]
    [string]$Device = 'CU129'
)

$ErrorActionPreference = 'Stop'
& (Join-Path (Split-Path $PSScriptRoot -Parent) 'install.ps1') -Device $Device
