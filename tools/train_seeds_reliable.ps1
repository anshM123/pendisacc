# Three seeds, identical config. Reliability is the claim being tested, so the
# protocol has to be repeated, not run once and reported.
#
# Launched through the Task Scheduler (see train_seeds_launch.cmd), NOT as a
# child of an interactive shell: the previous batch died mid-seed-2 when the
# shell that spawned it was torn down, because Start-Process children stay
# inside the parent's job object.
$ROOT = "C:\Users\anshm\Downloads\pendisaac"
foreach ($s in 1, 2, 3) {
    Write-Output "########## swing-up seed $s ##########"
    & "$ROOT\run.cmd" "$ROOT\experiments\train.py" --task TIP-SwingUp-v0 `
        --num_envs 4096 --max_iterations 1000 --seed $s --run_name "rel$s" --headless `
        *> "$ROOT\tools\train_swing_rel$s.txt"
    Write-Output "seed $s exit=$LASTEXITCODE"
}
# A file that only exists on success, so a waiter never blocks on log text that
# Kit may swallow.
Set-Content -Path "$ROOT\results\seeds_done.txt" -Value (Get-Date -Format o)
Write-Output "########## ALL SEEDS DONE ##########"
