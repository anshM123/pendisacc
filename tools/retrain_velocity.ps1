# Retrain both tasks after the actuator reversal to STEP/DIR position mode
# (RL action = cart velocity). The force-action policies are not reusable.
$ROOT = "C:\Users\anshm\Downloads\pendisaac"
Write-Output "########## balance (cheap check the action space is controllable) ##########"
& "$ROOT\run.cmd" "$ROOT\experiments\train.py" --task TIP-Balance-v0 `
    --num_envs 4096 --max_iterations 600 --seed 1 --run_name vel1 --headless `
    *> "$ROOT\tools\train_bal_vel.txt"
Write-Output "balance exit=$LASTEXITCODE"
Write-Output "########## swing-up ##########"
& "$ROOT\run.cmd" "$ROOT\experiments\train.py" --task TIP-SwingUp-v0 `
    --num_envs 4096 --seed 1 --run_name vel1 --headless `
    *> "$ROOT\tools\train_swing_vel.txt"
Write-Output "swingup exit=$LASTEXITCODE"
Write-Output "########## BOTH DONE ##########"
