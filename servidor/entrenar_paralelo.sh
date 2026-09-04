#!/usr/bin/env bash
# Se ejecuta EN EL SERVIDOR. Reparte UNA rejilla entre varias GPUs, un
# contenedor por GPU. No toca el contenedor 'stk' (el barrido secuencial).
#
#   bash servidor/entrenar_paralelo.sh --sizes 256 --bases 32 48 64
#
# Variables:
#   STK_GPUS="1 2 3 4 5 6 7"   que GPUs usar (por defecto 1..7, dejando la 0)
#   STK_IMAGE=imagen:tag       forzar imagen
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
mkdir -p logs salidas experimentos/resultados

DATA_ROOT="${STK_DATA_ROOT:-}"
if [ -z "$DATA_ROOT" ]; then
  for c in "$PROJ/dataset/dense_data" "$PROJ/dense_data" \
           /home/fsotoj/proyecto_stk/dataset/dense_data; do
    [ -d "$c" ] && { DATA_ROOT="$c"; break; }
  done
fi
[ -d "$DATA_ROOT" ] || { echo "ERROR: no encuentro dense_data" >&2; exit 1; }
echo "dataset : $DATA_ROOT"

MIN_FREE="${STK_MIN_FREE_MB:-10000}"
if [ -n "${STK_GPUS:-}" ]; then
  GPUS="$STK_GPUS"
  echo "gpus    : $GPUS (forzadas por STK_GPUS)"
else
  echo "memoria libre por GPU (se saltan las ocupadas y la 0, que usa el secuencial):"
  GPUS=""
  while IFS=, read -r idx free; do
    idx="$(echo $idx | tr -d ' ')"; free="$(echo $free | tr -d ' ')"
    if [ "$idx" = "0" ]; then
      echo "  gpu $idx: ${free} MiB libres  -> reservada al barrido secuencial"
    elif [ "$free" -ge "$MIN_FREE" ]; then
      echo "  gpu $idx: ${free} MiB libres  -> LIBRE"
      GPUS="$GPUS $idx"
    else
      echo "  gpu $idx: ${free} MiB libres  -> ocupada por otro proceso, se salta"
    fi
  done <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)"
fi
GPUS="$(echo $GPUS)"
[ -n "$GPUS" ] && [ "$GPUS" != " " ] || { echo "ERROR: no hay ninguna GPU libre (>= ${MIN_FREE} MiB)" >&2; exit 1; }
N=$(echo $GPUS | wc -w | tr -d " ")
echo "gpus    : $GPUS  ($N particiones)"

IMAGE="${STK_IMAGE:-}"
if [ -z "$IMAGE" ]; then
  for cand in $(docker images --format '{{.Repository}}:{{.Tag}}' | grep -v '<none>' \
                | grep -iE 'pytorch|torch|vllm|cuda|nvidia' | head -8); do
    if docker run --rm --entrypoint python3 "$cand" -c 'import torch' >/dev/null 2>&1; then
      IMAGE="$cand"; break
    fi
  done
fi
[ -n "$IMAGE" ] || { echo "ERROR: ninguna imagen local tiene torch" >&2; exit 1; }
echo "imagen  : $IMAGE"

TS="$(date +%Y%m%d_%H%M%S)"
i=0
for g in $GPUS; do
  NAME="stk_g$g"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" \
    --gpus all -e CUDA_VISIBLE_DEVICES="$g" \
    --ipc=host -u "$(id -u):$(id -g)" \
    -v "$PROJ:/work" -v "$DATA_ROOT:/data:ro" \
    -w /work -e HOME=/work -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint /bin/bash "$IMAGE" \
    -c "python3 experimentos/sweep.py --data-root /data --shard $i --num-shards $N \
          --epochs 40 --early-stop 12 --batch-size 16 --num-workers 6 \
          --out-dir /work/salidas/barrido_${TS}_g$g \
          --results-csv /work/experimentos/resultados/sweep_results_${TS}_g$g.csv \
          --runs-csv /work/experimentos/resultados/runs_${TS}_g$g.csv $* \
        2>&1 | tee /work/logs/par_${TS}_g$g.log" >/dev/null
  echo "  gpu $g -> contenedor $NAME (particion $i/$N)"
  i=$((i+1))
done

sleep 8
echo
echo "lanzadas $N particiones en paralelo. El barrido secuencial 'stk' sigue en la GPU 0."
docker ps --filter name=stk --format '  {{.Names}}\t{{.Status}}'
echo
echo "  progreso : bash servidor/estado.sh"
echo "  logs     : $PROJ/logs/par_${TS}_g*.log"
