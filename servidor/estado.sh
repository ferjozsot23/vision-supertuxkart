#!/usr/bin/env bash
# Se ejecuta EN LOCAL:  bash servidor/estado.sh
# Muestra el estado de lo que corre en el DGX. Solo lee.
SERVER=fsotoj@172.28.230.10
ssh "$SERVER" 'bash -s' <<'REMOTE'
P=/home/fsotoj/proyecto_stk
echo "== contenedores =="
docker ps -a --filter name=stk --format '  {{.Names}}\t{{.Status}}' 2>/dev/null || echo "  (ninguno)"

# ---- barrido secuencial (contenedor 'stk') ----
L="$(ls -t $P/logs/sweep_*.log 2>/dev/null | head -1)"
if [ -n "$L" ]; then
  TOT="$(grep -m1 -oE 'de [0-9]+ en la rejilla|BARRIDO: [0-9]+' "$L" | grep -oE '[0-9]+' | head -1)"
  echo; echo "== SECUENCIAL (gpu 0) — $(basename $L) =="
  echo "  terminadas: $(grep -c '^>>> ' "$L")/${TOT:-?}   epoca actual: $(grep -E '=== EPOCH' "$L" | tail -1)"
  grep -E '^>>> ' "$L" | sed 's/^/  /'
fi

# ---- particiones paralelas ----
PL="$(ls -t $P/logs/par_*_g*.log 2>/dev/null)"
if [ -n "$PL" ]; then
  TS="$(ls -t $P/logs/par_*_g*.log | head -1 | sed -E 's/.*par_([0-9_]+)_g[0-9]+\.log/\1/')"
  echo; echo "== PARALELO (lote $TS) =="
  for f in $P/logs/par_${TS}_g*.log; do
    g="$(echo $f | sed -E 's/.*_g([0-9]+)\.log/\1/')"
    # grep -c imprime 0 Y devuelve codigo 1; el '|| echo 0' de antes anadia un
    # segundo cero y la linea salia partida.
    n="$(grep -c '^>>> ' "$f" 2>/dev/null | head -1)"
    t="$(grep -m1 -oE '\-> [0-9]+ combinaciones' "$f" 2>/dev/null | grep -oE '[0-9]+' | head -1)"
    echo "  gpu $g: ${n:-0}/${t:-?}   $(grep -E '^>>> ' "$f" 2>/dev/null | tail -1)"
  done
fi

# ---- si alguna particion fallo, mostrar el error de verdad ----
for f in $P/logs/par_*_g*.log; do
  [ -f "$f" ] && grep -q 'ERROR\]' "$f" 2>/dev/null && {
    echo; echo "== FALLO en $(basename $f) =="
    grep -E 'Error|error|Traceback|raise |Exception|out of memory|Killed' "$f" | tail -12
  }
done

# ---- ranking si algo ya termino ----
for f in $P/logs/sweep_*.log $P/logs/par_*_g*.log; do
  [ -f "$f" ] && grep -q 'RESULTADOS DEL BARRIDO' "$f" 2>/dev/null && {
    echo; echo "== ranking de $(basename $f) =="; sed -n '/RESULTADOS DEL BARRIDO/,$p' "$f"; }
done
REMOTE
