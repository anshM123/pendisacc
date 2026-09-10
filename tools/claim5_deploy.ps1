# Deploy every Claim-5 policy into the frozen R*.
#
# Checkpoint selection is deliberate: each policy's best checkpoint is chosen by
# measured success in its OWN training simulator, which is what a practitioner
# with no access to reality would do. Selecting on R* would leak the answer.
$ROOT = "C:\Users\anshm\Downloads\pendisaac"
New-Item -ItemType Directory -Force -Path "$ROOT\results\claim5","$ROOT\results\claim5_select" | Out-Null
$plan = Get-Content "$ROOT\xi\claim5_plan.json" -Raw | ConvertFrom-Json

foreach ($sim in $plan.simulators.PSObject.Properties.Name) {
  $n = $plan.simulators.$sim.seeds
  for ($s = 1; $s -le $n; $s++) {
    $tag = "C5_${sim}_s$s"
    $dir = Get-ChildItem "$ROOT\logs\rsl_rl\tip_swingup" -Directory |
           Where-Object { $_.Name -like "*_$tag" } | Sort-Object Name | Select-Object -Last 1
    if (-not $dir) { "MISSING RUN: $tag"; continue }
    if (Test-Path "$ROOT\results\claim5\$tag.json") { "skip $tag"; continue }

    # 1. select the best checkpoint IN ITS OWN TRAINING SIMULATOR
    $sel = "$ROOT\results\claim5_select\$tag.json"
    if (-not (Test-Path $sel)) {
      & "$ROOT\run.cmd" "$ROOT\experiments\evaluate.py" --run $dir.FullName `
          --stride 200 --num_envs 256 --xi "$ROOT\xi\train\$sim.json" --out $sel `
          *> "$ROOT\tools\c5sel_$tag.txt"
    }
    if (-not (Test-Path $sel)) { "SELECT FAILED: $tag"; continue }
    $best = (Get-Content $sel -Raw | ConvertFrom-Json).results |
            Sort-Object -Property @{e={$_.success_rate}}, @{e={-$_.early_termination_rate}} -Descending |
            Select-Object -First 1
    $ckpt = Join-Path $dir.FullName $best.checkpoint
    "$tag -> $($best.checkpoint) (own-sim success $([math]::Round(100*$best.success_rate,1))%)"

    # 2. deploy into R*
    & "$ROOT\run.cmd" "$ROOT\experiments\evaluate.py" --checkpoint $ckpt `
        --num_envs 256 --xi "$ROOT\xi\R_star.json" --out "$ROOT\results\claim5\$tag.json" `
        *> "$ROOT\tools\c5dep_$tag.txt"
    if (-not (Test-Path "$ROOT\results\claim5\$tag.json")) { "DEPLOY FAILED: $tag" }
  }
}
Set-Content -Path "$ROOT\results\claim5_deploy_done.txt" -Value (Get-Date -Format o)
