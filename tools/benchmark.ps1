# Env-count scaling benchmark. Plan section 31 -- establishes the real compute
# cost of the project on this laptop before anything is promised to the 4090.
$env:OMNI_KIT_ACCEPT_EULA="YES"
$env:PYTHONUNBUFFERED="1"
$PY   = "C:\Users\anshm\Downloads\pendisaac\env_isaaclab\python.exe"
$ROOT = "C:\Users\anshm\Downloads\pendisaac"
$rows = @()
foreach ($n in @(64, 256, 1024, 2048, 4096)) {
    Write-Output "--- num_envs=$n ---"
    & $PY "$ROOT\tools\smoke_bounded.py" --num_envs $n --steps 100 *> "$ROOT\tools\bench_$n.log"
    $code = $LASTEXITCODE
    $res  = "$ROOT\results\smoke_result.json"
    if ($code -eq 0 -and (Test-Path $res)) {
        $j = Get-Content $res -Raw | ConvertFrom-Json
        $rows += [pscustomobject]@{ num_envs=$n; status="PASS"; seconds=[math]::Round($j.seconds,2); env_steps_per_s=[math]::Round($j.env_steps_per_s,0) }
        Copy-Item $res "$ROOT\results\smoke_result_$n.json" -Force
    } else {
        $rows += [pscustomobject]@{ num_envs=$n; status="FAIL(exit=$code)"; seconds=$null; env_steps_per_s=$null }
    }
}
$rows | Format-Table -AutoSize
$rows | Export-Csv "$ROOT\results\benchmark_laptop.csv" -NoTypeInformation
Write-Output "=== wrote results\benchmark_laptop.csv ==="
