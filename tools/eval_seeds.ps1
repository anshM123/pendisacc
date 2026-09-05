# Evaluate every seed of the reliability batch by MEASURED success, then print
# one table. Reward is not the metric -- see experiments/evaluate.py.
$ROOT = "C:\Users\anshm\Downloads\pendisaac"
foreach ($s in 1, 2, 3) {
    $run = Get-ChildItem "$ROOT\logs\rsl_rl\tip_swingup" -Directory |
           Where-Object { $_.Name -like "*_rel$s" } |
           Sort-Object Name | Select-Object -Last 1
    if (-not $run) { Write-Output "seed ${s}: no run directory"; continue }
    Write-Output "########## evaluating seed $s : $($run.Name) ##########"
    & "$ROOT\run.cmd" "$ROOT\experiments\evaluate.py" --run $run.FullName `
        --stride 100 --num_envs 512 --out "$ROOT\results\eval_rel$s.json" `
        *> "$ROOT\tools\eval_rel$s.txt"
    Write-Output "seed $s exit=$LASTEXITCODE"
}
Set-Content -Path "$ROOT\results\eval_done.txt" -Value (Get-Date -Format o)
