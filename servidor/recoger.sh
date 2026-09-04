#!/usr/bin/env bash
# Se ejecuta EN LOCAL. Trae modelo, bitacora y logs desde el DGX.
set -euo pipefail

# Descarga a la raiz del proyecto, se invoque desde donde se invoque.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER=fsotoj@172.28.230.10
REMOTE=/home/fsotoj/proyecto_stk
mkdir -p logs salidas experimentos/resultados
# Todos los modelos, no solo model.th: el mejor hasta ahora es model_256.th y
# se estaba quedando en el servidor.
rsync -avz "$SERVER:$REMOTE/salidas/"model*.th ./salidas/ 2>/dev/null || echo "(aun no hay modelos)"
rsync -avz "$SERVER:$REMOTE/experimentos/resultados/runs.csv" ./experimentos/resultados/ 2>/dev/null || echo "(aun no hay runs.csv)"
# Resultados del barrido secuencial y de todas las particiones paralelas.
rsync -avz "$SERVER:$REMOTE/experimentos/resultados/"sweep_results*.csv ./experimentos/resultados/ 2>/dev/null || true
rsync -avz "$SERVER:$REMOTE/salidas/"barrido_*/ ./salidas/barridos/ --include='*/' --include='*.th' --exclude='*' 2>/dev/null || true
rsync -avz "$SERVER:$REMOTE/salidas/barrido/" ./salidas/barrido/ --include='*/' --include='*.th' --exclude='*' 2>/dev/null || true
rsync -avz "$SERVER:$REMOTE/logs/" ./logs/
echo
echo "==> ultima tabla de IoU:"
grep -B1 -A13 '=== EPOCH' logs/*.log 2>/dev/null | tail -20
