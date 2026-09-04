#!/usr/bin/env bash
# Se ejecuta EN LOCAL. Sube dataset (si hace falta) + codigo al DGX y lanza el
# entrenamiento. Un solo comando:  bash servidor/lanzar.sh
#
# Cualquier flag extra se le pasa a train.py:
#   bash servidor/lanzar.sh --size 256 256 --notes "corrida 2"
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVER=fsotoj@172.28.230.10
REMOTE=/home/fsotoj/proyecto_stk

CTL="$HOME/.ssh/cm_stk_$$"
mkdir -p "$HOME/.ssh"
SSH=(ssh -o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=10m)
trap '"${SSH[@]}" -O exit "$SERVER" 2>/dev/null || true' EXIT
export RSYNC_RSH="ssh -o ControlMaster=auto -o ControlPath=$CTL -o ControlPersist=10m"

"${SSH[@]}" "$SERVER" "mkdir -p '$REMOTE'"

if "${SSH[@]}" "$SERVER" "[ -d '$REMOTE/dataset/dense_data' ]"; then
  echo "==> el dataset ya esta en el servidor, no lo subo"
else
  echo "==> el dataset no esta en el servidor: subiendo dataset.zip (327 MB)"
  echo "    (se sube el zip y se descomprime alla: un archivo grande viaja"
  echo "     mucho mas rapido que 22.000 archivos pequenos)"
  rsync -avz --progress dataset.zip "$SERVER:$REMOTE/"
  echo "==> descomprimiendo en el servidor"
  "${SSH[@]}" "$SERVER" "cd '$REMOTE' && \
      (command -v unzip >/dev/null && unzip -q -o dataset.zip -x '__MACOSX/*' '*.DS_Store' -d . \
       || python3 -c \"
import zipfile
z = zipfile.ZipFile('dataset.zip')
for n in z.namelist():
    if n.startswith('__MACOSX/') or n.endswith('.DS_Store'):
        continue
    z.extract(n, '.')
\") && rm -f dataset.zip && \
      echo \"   tracks: \$(ls dataset/dense_data | tr '\n' ' ')\""
fi

echo "==> subiendo codigo"
rsync -avz --exclude dataset --exclude 'dataset.zip' --exclude __pycache__ \
      --exclude salidas --exclude logs --exclude ejemplos --exclude '*.th' --exclude .DS_Store \
      --exclude '*.pdf' --exclude .claude --exclude .git --exclude docs \
      ./ "$SERVER:$REMOTE/"

echo "==> lanzando entrenamiento"
LAUNCHER=servidor/entrenar.sh
if [ "${1:-}" = "paralelo" ]; then LAUNCHER=servidor/entrenar_paralelo.sh; shift; fi
RARGS=""
for a in "$@"; do RARGS="$RARGS $(printf '%q' "$a")"; done
"${SSH[@]}" "$SERVER" "bash '$REMOTE/$LAUNCHER'$RARGS"

cat <<EOF

Listo. El entrenamiento corre desacoplado en el servidor; podes cerrar la terminal.

  ver progreso :  bash servidor/estado.sh
  en vivo      :  ssh $SERVER 'docker logs -f stk'
  traer todo   :  bash servidor/recoger.sh

EOF
