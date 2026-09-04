# ==============================================================================
# FROZEN STACK — triple inverted pendulum sim2real project
#   Python      3.11.16   (Isaac Sim 5.1.0 ships cp311 wheels ONLY)
#   Isaac Sim   5.1.0     (pip, win_amd64)
#   PyTorch     2.7.0+cu128  (cu128 required for Blackwell / RTX 50-series sm_120)
#   Isaac Lab   2.3.2     (source, editable, from C:\Users\anshm\Downloads\IsaacLab-main)
#   RL          rsl-rl 5.0.1 (+ skrl, sb3 for baselines; rl-games skipped: needs git+gym)
# ==============================================================================
$ErrorActionPreference = "Stop"

$ROOT     = "C:\Users\anshm\Downloads\pendisaac"
$VENV     = "$ROOT\env_isaaclab"
$PY       = "$VENV\python.exe"
$UV       = "$ROOT\tools\uv.exe"
$ISAACLAB = "C:\Users\anshm\Downloads\IsaacLab-main\IsaacLab-main"

function Step($n, $msg) { Write-Output "`n=== [$n] $msg ===" }

Step 1 "Build backend pins (setuptools<82 to avoid pkg_resources breakage)"
& $UV pip install --python $PY "setuptools<82.0.0" wheel toml
if (-not $?) { throw "setuptools pin failed" }

Step 2 "Isaac Sim 5.1.0 -- the big one (~15 GB, expect 20-40 min)"
& $UV pip install --python $PY "isaacsim[all,extscache]==5.1.0" `
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
if (-not $?) { throw "isaacsim install failed" }

Step 3 "PyTorch 2.7.0 + cu128"
& $UV pip install --python $PY -U torch==2.7.0 torchvision==0.22.0 `
    --index-url https://download.pytorch.org/whl/cu128
if (-not $?) { throw "torch install failed" }

Step 4 "Isaac Lab 2.3.2 extensions (editable)"
foreach ($d in @("isaaclab","isaaclab_assets","isaaclab_contrib","isaaclab_mimic","isaaclab_tasks")) {
    Write-Output "  -> $d"
    & $UV pip install --python $PY -e "$ISAACLAB\source\$d"
    if (-not $?) { throw "$d install failed" }
}

Step 5 "isaaclab_rl with rsl-rl / skrl / sb3"
& $UV pip install --python $PY -e "$ISAACLAB\source\isaaclab_rl[rsl-rl,skrl,sb3]"
if (-not $?) { throw "isaaclab_rl install failed" }

Step 6 "Re-assert torch (extensions can drag in a CPU build)"
& $UV pip install --python $PY -U torch==2.7.0 torchvision==0.22.0 `
    --index-url https://download.pytorch.org/whl/cu128
if (-not $?) { throw "torch re-assert failed" }

Step 7 "Analysis-side deps (nonlinear dynamics / stats / IO / meshes)"
& $UV pip install --python $PY "numpy<2" scipy sympy pandas pyarrow matplotlib `
    tqdm rich pyyaml scikit-learn trimesh networkx
if (-not $?) { throw "analysis deps failed" }

Step 8 "Verify"
& $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
& $PY -c "import isaacsim, isaaclab, isaaclab_tasks, rsl_rl; print('isaaclab OK')"

Write-Output "`n=== INSTALL COMPLETE ==="
