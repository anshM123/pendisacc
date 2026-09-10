# Claim 5: train a policy in each simulator, deploy all into the common R*.
# Pre-registered in PREREGISTRATION_H3.md (commit 3d453a0) before any run.
$ROOT = "C:\Users\anshm\Downloads\pendisaac"
$plan = Get-Content "$ROOT\xi\claim5_plan.json" -Raw | ConvertFrom-Json
foreach ($name in $plan.simulators.PSObject.Properties.Name) {
    $n = $plan.simulators.$name.seeds
    for ($s = 1; $s -le $n; $s++) {
        $tag = "C5_${name}_s$s"
        if (Get-ChildItem "$ROOT\logs\rsl_rl\tip_swingup" -Directory -EA SilentlyContinue |
            Where-Object { $_.Name -like "*_$tag" }) { "skip $tag"; continue }
        "### $tag"
        & "$ROOT\run.cmd" "$ROOT\experiments\train.py" --task TIP-SwingUp-v0 `
            --num_envs 4096 --max_iterations 1000 --seed $s --run_name $tag --headless `
            --xi "$ROOT\xi\train\$name.json" *> "$ROOT\tools\train_$tag.txt"
        "$tag exit=$LASTEXITCODE"
    }
}
Set-Content -Path "$ROOT\results\claim5_training_done.txt" -Value (Get-Date -Format o)
