# Build vendor-native HIP and Intel Level Zero plugins on Windows x64.
# Requires an installed AMD HIP SDK and Intel oneAPI DPC++ compiler.
# Run in a developer PowerShell with both compiler toolchains initialized.
# No GPU device is required for compilation; device execution is validated on
# actual vendor hardware in a separate parity/benchmark job.
param(
    [ValidateSet('hip', 'level-zero', 'all')]
    [string]$Backend = 'all',
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    if ($env:BATO_NATIVE_OUT) {
        $OutputDirectory = $env:BATO_NATIVE_OUT
    } else {
        $OutputDirectory = Join-Path $Root 'target/native'
    }
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path $OutputDirectory).Path

function Find-Compiler {
    param([string[]]$Names, [string]$Description)
    foreach ($Name in $Names) {
        $Found = Get-Command $Name -ErrorAction SilentlyContinue
        if ($Found) { return $Found.Source }
    }
    throw "$Description compiler not found. Install the vendor SDK and initialize its environment before running this script."
}
function Invoke-Compiler {
    param([string]$Compiler, [string[]]$Arguments)
    & $Compiler @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Compiler $Compiler exited with code $LASTEXITCODE"
    }
}
function Build-Hip {
    $Hipcc = Find-Compiler @('hipcc.exe', 'hipcc.bat', 'hipcc') 'AMD HIP'
    $Source = Join-Path $Root 'native/hip/filters.hip'
    $Output = Join-Path $OutputDirectory 'bato_hip.dll'
    $Args = @('-O3', '-std=c++17', '-shared', '-x', 'hip')
    if ($env:BATO_HIP_OFFLOAD_ARCHS) {
        foreach ($Arch in ($env:BATO_HIP_OFFLOAD_ARCHS -split '[,; ]+' | Where-Object { $_ })) {
            if ($Arch -notmatch '^gfx[0-9a-f]+$') {
                throw "Invalid BATO_HIP_OFFLOAD_ARCHS entry: $Arch"
            }
            $Args += "--offload-arch=$Arch"
        }
    }
    $Args += @($Source, '-o', $Output)
    Invoke-Compiler $Hipcc $Args
    if (-not (Test-Path $Output -PathType Leaf)) { throw "HIP SDK did not produce $Output" }
    Write-Host "Built native AMD HIP DLL: $Output"
}
function Build-LevelZero {
    $Icx = Find-Compiler @('icx.exe', 'icpx.exe', 'icx', 'icpx') 'Intel oneAPI DPC++'
    $Source = Join-Path $Root 'native/level_zero/filters.cpp'
    $Output = Join-Path $OutputDirectory 'bato_level_zero.dll'
    Invoke-Compiler $Icx @('-O3', '-std=c++17', '-fsycl', '-fsycl-targets=spir64', '-shared', $Source, '-o', $Output)
    if (-not (Test-Path $Output -PathType Leaf)) { throw "oneAPI did not produce $Output" }
    Write-Host "Built native Intel Level Zero DLL: $Output"
}

switch ($Backend) {
    'hip' { Build-Hip }
    'level-zero' { Build-LevelZero }
    'all' {
        Build-Hip
        Build-LevelZero
    }
}
