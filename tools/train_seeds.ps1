# Multi-seed batch. The plan requires several seeds per simulator so that
# transfer differences can be attributed to physics rather than to PPO variance.
# Reward plateaus by ~iteration 160, so 600 iterations is ample.
$ROOT = "C:\Users\anshm\Downloads\pendisaac"
foreach ($s in 1, 2, 3) {
    Write-Output "########## seed $s ##########"
    & "$ROOT\run.cmd" "$ROOT\experiments\train.py" --task TIP-Balance-v0 `
        --num_envs 4096 --max_iterations 600 --seed $s --run_name "v2s$s" --headless `
        *> "$ROOT\tools\train_v2s$s.txt"
    Write-Output "seed $s exit=$LASTEXITCODE"
}
Write-Output "########## BATCH DONE ##########"
