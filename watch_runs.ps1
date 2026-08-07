# watch_runs.ps1 -- live status for the HeatGNN baseline queue and any train.lodo run.
# Read-only: counts artifacts, tails logs, asks nvidia-smi. Touches nothing.
#   .\watch_runs.ps1              refresh every 30s until Ctrl+C
#   .\watch_runs.ps1 -Once        print one snapshot and exit
#   .\watch_runs.ps1 -Every 120   refresh every 2 minutes
param([switch]$Once, [int]$Every = 30)

$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }

function Show-Status {
    $now = Get-Date

    # --- HeatGNN: completions are npz files; the runner decides remaining work from these ---
    $done = @(Get-ChildItem "$root\baselines\_preds" -Filter 'HeatGNN*.npz' -ErrorAction SilentlyContinue)
    $byds = $done | Group-Object { ($_.BaseName -split '__')[1] } | ForEach-Object { "$($_.Name.Replace('influenza_','')) $($_.Count)" }
    Write-Host ("[{0:HH:mm:ss}]  HeatGNN {1}/60   ({2})" -f $now, $done.Count, ($byds -join ', ')) -ForegroundColor Cyan
    if ($done.Count) {
        $newest = $done | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $mins = [math]::Round(($now - $newest.LastWriteTime).TotalMinutes)
        Write-Host ("           last completion {0} min ago: {1}" -f $mins, $newest.BaseName)
    }

    # --- live workers: epoch + val_loss from each log (needs the per-epoch lines) ---
    # 60-min window, not 15: logs are BLOCK-BUFFERED (export_heatgnn's _cmd has no `python -u`), so a
    # live worker can go many minutes between flushes. Same reason the epoch shown lags the real one,
    # and why epoch deltas between two refreshes are not a reliable per-epoch rate.
    $logs = @(Get-ChildItem "$root\baselines\_logs" -Filter 'HeatGNN__*.log' -ErrorAction SilentlyContinue |
              Where-Object { $_.LastWriteTime -gt $now.AddMinutes(-60) } |
              Sort-Object LastWriteTime -Descending | Select-Object -First 8)
    foreach ($lg in $logs) {
        $line = (Select-String -Path $lg.FullName -Pattern '^Epoch' | Select-Object -Last 1).Line
        if ($line -match 'Epoch (\d+).*?val_loss ([\d.eE+-]+)') {
            Write-Host ("           {0,-44} ep {1,5}  val {2}" -f $lg.BaseName.Replace('HeatGNN__',''), $Matches[1], $Matches[2])
        }
    }
    if (-not $logs.Count) { Write-Host "           no log written in the last 15 min" -ForegroundColor DarkYellow }

    # --- LODO / LDO: artifacts only appear when a whole fold finishes ---
    $ldo  = @(Get-ChildItem "$root\results\lodo" -Filter 'encoder_ldo*' -ErrorAction SilentlyContinue)
    $proc = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
              Where-Object { $_.CommandLine -like '*train.lodo*' })
    $tag = if ($proc.Count) {
        $p = Get-Process -Id $proc[0].ProcessId -ErrorAction SilentlyContinue
        $args = ($proc[0].CommandLine -split 'train\.lodo')[-1].Trim()
        "RUNNING ({0}), up {1:N2} h" -f $args, ($now - $p.StartTime).TotalHours
    } else { 'no process' }
    Write-Host ("           LDO artifacts {0}   [{1}]" -f $ldo.Count, $tag) -ForegroundColor Cyan

    # --- GPU (LODO is GPU, HeatGNN is CPU, so this tracks the LODO run) ---
    $gpu = (nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>$null)
    if ($gpu) { Write-Host "           GPU $gpu" }
    Write-Host ''
}

if ($Once) { Show-Status; return }
while ($true) { Show-Status; Start-Sleep -Seconds $Every }
