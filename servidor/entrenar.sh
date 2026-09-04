#!/usr/bin/env bash
# Se ejecuta EN EL SERVIDOR (DGX-H200-USFQ). Localiza el dataset, resuelve donde
# vive torch (python nativo o, si no, un contenedor Docker) y lanza el
# entrenamiento desacoplado de la sesion ssh.
#
#   bash servidor/entrenar.sh                    # baseline 128x128, 40 epocas
#   bash servidor/entrenar.sh --size 256 256     # los flags extra van a train.py
#
# Variables opcionales:
#   STK_DATA_ROOT=/ruta/dense_data   saltarse la deteccion del dataset
#   STK_IMAGE=imagen:tag             forzar la imagen de Docker
#   STK_GPU=3                        usar otra GPU (por defecto la 0)
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
mkdir -p logs salidas experimentos/resultados

DATA_ROOT="${STK_DATA_ROOT:-}"
if [ -z "$DATA_ROOT" ]; then
  for c in "$PROJ/dataset/dense_data" "$PROJ/dense_data" \
           /home/fsotoj/proyecto_stk/dataset/dense_data \
           /home/fsotoj/dataset/dense_data /home/fsotoj/dense_data; do
    [ -d "$c" ] && { DATA_ROOT="$c"; break; }
  done
fi
if [ -z "$DATA_ROOT" ]; then
  echo "buscando dense_data bajo /home/fsotoj (max 60s)..."
  DATA_ROOT="$(timeout 60 find /home/fsotoj -maxdepth 5 -type d -name dense_data 2>/dev/null | head -1 || true)"
fi
if [ -z "$DATA_ROOT" ] || [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: no encuentro 'dense_data'. Pasalo a mano:" >&2
  echo "  STK_DATA_ROOT=/ruta/a/dense_data bash servidor/entrenar.sh" >&2
  exit 1
fi
NTRACKS="$(find "$DATA_ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l)"
echo "dataset : $DATA_ROOT  ($NTRACKS tracks)"
[ "$NTRACKS" -ge 2 ] || { echo "ERROR: hacen falta >=2 tracks para el split" >&2; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
GPU="${STK_GPU:-0}"

if [ "${1:-}" = "sweep" ]; then
  shift
  SCRIPT=experimentos/sweep.py
  LOG="$PROJ/logs/sweep_$TS.log"
  DEFAULTS=(--epochs 40 --early-stop 12 --batch-size 16 --num-workers 8
            --out-dir OUTDIR/salidas/barrido
            --results-csv OUTDIR/experimentos/resultados/sweep_results.csv)
else
  SCRIPT=train.py
  LOG="$PROJ/logs/run_$TS.log"
  DEFAULTS=(--epochs 40 --size 128 128 --lr 1e-3 --batch-size 16 --num-workers 8
            --out OUTDIR/salidas/model.th --ckpt-dir OUTDIR/salidas/checkpoints)
fi
echo "script  : $SCRIPT"

PY="$(command -v python3 || command -v python || true)"
if [ -n "$PY" ] && "$PY" -c 'import torch' >/dev/null 2>&1; then
  echo "modo    : python nativo ($PY)"
  "$PY" -c "import torch;print('torch   :',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.device_count(),'gpu(s)')"
  tmux kill-session -t stk 2>/dev/null || true
  tmux new-session -d -s stk \
    "cd '$PROJ' && CUDA_VISIBLE_DEVICES=$GPU '$PY' $SCRIPT --data-root '$DATA_ROOT' \
       $(printf '%q ' "${DEFAULTS[@]//OUTDIR/$PROJ}") --runs-csv '$PROJ/experimentos/resultados/runs.csv' \
       $(printf '%q ' "$@") 2>&1 | tee '$LOG'"
  echo; echo "lanzado en tmux 'stk'   log: $LOG"
  echo "  seguir : tmux attach -t stk      (salir sin matar: Ctrl+B, luego D)"
  exit 0
fi

command -v docker >/dev/null || { echo "ERROR: sin torch nativo y sin docker" >&2; exit 1; }
echo "modo    : docker (el python del sistema no tiene torch)"

IMAGE="${STK_IMAGE:-}"
if [ -z "$IMAGE" ]; then
  echo "buscando una imagen local que traiga torch..."
  CANDS="$(docker images --format '{{.Repository}}:{{.Tag}}' \
           | grep -v '<none>' \
           | grep -iE 'pytorch|torch|vllm|cuda|nvidia|nemo|docling' | head -8)"
  for cand in $CANDS; do
    printf '  %-52s ' "$cand"
    V="$(docker run --rm --entrypoint python3 "$cand" -c 'import torch;print(torch.__version__)' 2>/dev/null | tail -1)"
    if [ -n "$V" ]; then echo "torch $V  <-- elegida"; IMAGE="$cand"; break; else echo "sin torch"; fi
  done
fi
[ -n "$IMAGE" ] || { echo "ERROR: ninguna imagen local tiene torch. Forzala con STK_IMAGE=..." >&2; exit 1; }
echo "imagen  : $IMAGE"
echo "gpu     : $GPU (de 8; se limita con CUDA_VISIBLE_DEVICES para no acaparar la maquina)"

docker rm -f stk >/dev/null 2>&1 || true
if ! docker run -d --name stk \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPU" \
  --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$PROJ:/work" -v "$DATA_ROOT:/data:ro" \
  -w /work -e HOME=/work -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint /bin/bash "$IMAGE" \
  -c "python3 -c \"
import torch;print('torch   :',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.device_count(),'gpu(s)')
try:
    import torchvision;print('tvision :',torchvision.__version__)
except ImportError:
    print('tvision : NO INSTALADO')
import socket
try:
    socket.create_connection(('download.pytorch.org',443),4);print('internet: si')
except Exception as e:
    print('internet: NO (%s)'%type(e).__name__)
\" && \
       python3 $SCRIPT --data-root /data $(printf '%q ' "${DEFAULTS[@]//OUTDIR//work}") \
         --runs-csv /work/experimentos/resultados/runs.csv $(printf '%q ' "$@") \
       2>&1 | tee /work/logs/$(basename "$LOG")"
then
  echo >&2
  echo "ERROR: 'docker run' fallo. Causas tipicas:" >&2
  echo "  - falta el runtime de NVIDIA: prueba a sustituir '--gpus all' por '--runtime=nvidia'" >&2
  echo "  - el usuario no esta en el grupo docker: 'groups' deberia incluir 'docker'" >&2
  exit 1
fi

sleep 6
echo
echo "lanzado en el contenedor 'stk'"
echo "  log      : $LOG"
echo "  seguir   : docker logs -f stk"
echo "  estado   : docker ps --filter name=stk"
echo "  parar    : docker rm -f stk"
echo
echo "--- primeras lineas ---"
docker logs stk 2>&1 | head -25
