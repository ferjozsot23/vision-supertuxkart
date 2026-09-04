#!/usr/bin/env bash
# Se ejecuta EN LOCAL. Trae modelo, bitacora y logs desde el DGX.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER=fsotoj@172.28.230.10
REMOTE=/home/fsotoj/proyecto_stk
mkdir -p logs salidas experimentos/resultados
rsync -avz "$SERVER:$REMOTE/salidas/"model*.th ./salidas/ 2>/dev/null || echo "(aun no hay modelos)"
rsync -avz "$SERVER:$REMOTE/experimentos/resultados/runs.csv" ./experimentos/resultados/ 2>/dev/null || echo "(aun no hay runs.csv)"
rsync -avz "$SERVER:$REMOTE/experimentos/resultados/"sweep_results*.csv ./experimentos/resultados/ 2>/dev/null || true
rsync -avz "$SERVER:$REMOTE/salidas/"barrido_*/ ./salidas/barridos/ --include='*/' --include='*.th' --exclude='*' 2>/dev/null || true
rsync -avz "$SERVER:$REMOTE/salidas/barrido/" ./salidas/barrido/ --include='*/' --include='*.th' --exclude='*' 2>/dev/null || true
rsync -avz "$SERVER:$REMOTE/logs/" ./logs/
echo
echo "==> ultima tabla de IoU:"
grep -B1 -A13 '=== EPOCH' logs/*.log 2>/dev/null | tail -20
