#!/usr/bin/env bash
# Se ejecuta EN LOCAL:  bash servidor/diagnostico.sh
# Averigua donde vive torch en el DGX. Solo lee, no cambia nada.
SERVER=fsotoj@172.28.230.10
ssh "$SERVER" 'bash -s' <<'REMOTE'
echo "================ HOST ================"
hostname; whoami; echo "HOME=$HOME"; uname -r

echo; echo "================ GPUs ================"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total --format=csv || echo "(no hay nvidia-smi)"

echo; echo "================ PYTHONS ================"
for p in python python3 python3.10 python3.11 python3.12; do
  q="$(command -v $p 2>/dev/null)" && echo "$p -> $q ($($p -V 2>&1))"
done

echo; echo "================ TORCH en los pythons del PATH ================"
for p in $(command -v python python3 2>/dev/null); do
  echo -n "$p : "
  "$p" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count(),'gpu(s)')" 2>&1 | head -1
done

echo; echo "================ CONDA / VENV ================"
for c in /opt/conda/bin/conda "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
  [ -x "$c" ] && { echo "conda: $c"; "$c" env list 2>/dev/null; }
done
ls -d /opt/conda/envs/* "$HOME"/*/bin/activate "$HOME"/*/*/bin/activate 2>/dev/null | head -20
echo "--- torch dentro de cada env de conda ---"
for py in /opt/conda/envs/*/bin/python "$HOME"/miniconda3/envs/*/bin/python "$HOME"/anaconda3/envs/*/bin/python; do
  [ -x "$py" ] && { echo -n "$py : "; "$py" -c "import torch;print(torch.__version__)" 2>&1|head -1; }
done

echo; echo "================ MODULES (Lmod / environment-modules) ================"
if command -v module >/dev/null 2>&1; then module avail 2>&1 | head -40; else echo "(no hay 'module')"; fi

echo; echo "================ CONTENEDORES / SCHEDULER ================"
for t in docker enroot srun sbatch sinfo singularity apptainer pyxis; do
  command -v $t >/dev/null && echo "hay: $t ($(command -v $t))"
done
command -v docker  >/dev/null && (docker images 2>/dev/null | head -15 || echo "(docker sin permisos)")
command -v enroot  >/dev/null && (enroot list 2>/dev/null | head -15)
command -v sinfo   >/dev/null && sinfo 2>/dev/null | head -10

echo; echo "================ PIP ================"
command -v pip3 >/dev/null && pip3 list 2>/dev/null | grep -iE "^(torch|numpy|pillow|matplotlib)" || echo "(pip3 sin torch)"

echo; echo "================ FIN ================"
REMOTE
