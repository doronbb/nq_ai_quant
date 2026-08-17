# ===========================================================================
#  Crash-restarting supervisor for the search loop (PowerShell version).
#  Unlike run_forever.bat this needs no console, so it can be launched
#  detached / in the background:
#      powershell -ExecutionPolicy Bypass -File run_forever.ps1 -Workers 3
#
#  -Workers N runs N search processes against the same results/ folder. That is
#  safe by design: the registry is SQLite in WAL mode and reserve() is atomic,
#  so two workers cannot claim the same genome.
#
#  Size N against model.threads in config.yaml, NOT against core count alone:
#  N x threads should land near your total hardware threads. Workers each
#  grabbing every core is slower than one worker, not faster.
#
#  Stop everything with:  Get-Process python | Stop-Process -Force
#  (and close this script, or it will restart them)
# ===========================================================================
param([int]$Workers = 3)

Set-Location $PSScriptRoot

# Resolve the real interpreter. On Windows `python` is often the WindowsApps
# launcher stub, which spawns the actual python and exits immediately — so a
# supervisor watching the stub sees every worker die a moment after it starts
# and restarts it forever.
$py = & python -c "import sys; print(sys.executable)"
if (-not $py) { $py = "python" }
Write-Output "[$(Get-Date -Format s)] supervising $Workers search worker(s) using $py"

$procs = @{}
while ($true) {
    for ($i = 0; $i -lt $Workers; $i++) {
        $alive = $procs.ContainsKey($i) -and -not $procs[$i].HasExited
        if (-not $alive) {
            if ($procs.ContainsKey($i)) {
                Write-Output "[$(Get-Date -Format s)] worker $i exited (code $($procs[$i].ExitCode)), restarting"
            }
            $procs[$i] = Start-Process -FilePath $py `
                -ArgumentList "run_search.py", "--config", "config.yaml" `
                -PassThru -WindowStyle Hidden
            Write-Output "[$(Get-Date -Format s)] worker $i started (pid $($procs[$i].Id))"
            Start-Sleep -Seconds 3      # stagger the feature-matrix builds
        }
    }
    Start-Sleep -Seconds 20
}
