# Run a measurement program and sample its GPU memory (#458, step E1).
#
# Windows reports the GPU memory of each process in the "GPU Process Memory" performance counters
# (the values that Task Manager shows). The script starts the program, samples the dedicated and
# shared usage of its process every -IntervalMs, and writes the peak values and the samples to
# -Out. The program output goes to -Log.
#
# Example:
#   .\run_with_gpu_memory.ps1 -Exe ..\..\..\rust\puerh_mind\target\release\examples\segmentation_runtime_report.exe `
#     -Arguments @('runtime', '--onnx-dir', 'D:\...\onnx\tiny', '--device', 'directml', ...) `
#     -Out gpu_memory_tiny.json -Log runtime_tiny.log

param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$Out,
    [Parameter(Mandatory = $true)][string]$Log,
    [int]$IntervalMs = 250
)

$ErrorActionPreference = 'Stop'

function Quote([string]$Value) {
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"', '\"') + '"' }
    return $Value
}

$process = Start-Process -FilePath $Exe -ArgumentList (($Arguments | ForEach-Object { Quote $_ }) -join ' ') `
    -NoNewWindow -PassThru -RedirectStandardOutput $Log -RedirectStandardError "$Log.stderr"
$samples = New-Object System.Collections.Generic.List[object]
$started = Get-Date
$peakDedicated = 0.0
$peakShared = 0.0
$counters = @(
    "\GPU Process Memory(pid_$($process.Id)_*)\Dedicated Usage",
    "\GPU Process Memory(pid_$($process.Id)_*)\Shared Usage"
)

while (-not $process.HasExited) {
    try {
        $values = (Get-Counter -Counter $counters -ErrorAction Stop).CounterSamples
        $dedicated = ($values | Where-Object { $_.Path -like '*dedicated usage' } | Measure-Object CookedValue -Sum).Sum
        $shared = ($values | Where-Object { $_.Path -like '*shared usage' } | Measure-Object CookedValue -Sum).Sum
        $peakDedicated = [Math]::Max($peakDedicated, $dedicated)
        $peakShared = [Math]::Max($peakShared, $shared)
        $samples.Add([ordered]@{
                seconds        = [Math]::Round(((Get-Date) - $started).TotalSeconds, 2)
                dedicated_bytes = $dedicated
                shared_bytes   = $shared
            })
    }
    catch {
        # The counter instance appears only after the process opens the GPU.
    }
    Start-Sleep -Milliseconds $IntervalMs
}
$process.WaitForExit()

$report = [ordered]@{
    source               = 'Windows performance counters: GPU Process Memory (Dedicated Usage, Shared Usage)'
    exit_code            = $process.ExitCode
    interval_ms          = $IntervalMs
    peak_dedicated_bytes = $peakDedicated
    peak_shared_bytes    = $peakShared
    sample_count         = $samples.Count
    samples              = $samples
}
$report | ConvertTo-Json -Depth 4 | Set-Content -Path $Out -Encoding utf8
Write-Output ("exit {0}; peak dedicated {1:N0} MB; peak shared {2:N0} MB; {3} samples" -f `
        $process.ExitCode, ($peakDedicated / 1MB), ($peakShared / 1MB), $samples.Count)
exit $process.ExitCode
