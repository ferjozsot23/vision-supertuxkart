# Experimentos

Herramientas para explorar hiperparámetros y analizar los resultados. Nada de
esto hace falta para *usar* el modelo — para eso está `predict.py` en la raíz.

## Barrido

`sweep.py` recorre el producto cartesiano de los ejes que se le den y escribe una
fila por combinación. Una combinación que falle no tumba el barrido: se registra
como `ERROR` y sigue con la siguiente.

```bash
python experimentos/sweep.py --data-root <ruta>/dense_data \
    --sizes 256 --bases 32 64 --lrs 1e-3 3e-4 --powers 0.25 0.30
```

Con `--num-shards N --shard I` procesa solo una de cada N combinaciones, lo que
permite repartir la misma rejilla entre varias GPUs lanzando N procesos. El
reparto es intercalado, no por bloques, para que las combinaciones caras no se
amontonen en la misma GPU. `servidor/entrenar_paralelo.sh` automatiza eso.

## Análisis

| script | para qué |
|---|---|
| `comparar.py` | mejor época de cada corrida de `runs.csv`, con el IoU por clase |
| `consolidar.py` | une los `sweep_results*.csv` de las particiones y ordena por mIoU |
| `evaluar_modelos.py` | vuelve a medir uno o varios `.th` sobre validación |

`consolidar.py` conserva la mejor de las repeticiones de una misma configuración
**y reporta el rango**, porque esa diferencia es la medida directa de la varianza
entre corridas — que en este problema resultó ser de ±0.04 de mIoU.

`evaluar_modelos.py` existe porque los CSV pueden mentir: al recoger resultados de
varias particiones, dos archivos con el mismo nombre pueden pisarse. Recalcular la
matriz de confusión sobre los 500 frames de validación es la única comprobación
que no depende de la contabilidad.

## resultados/

| archivo | qué contiene |
|---|---|
| `runs.csv` | una fila por época de cada corrida |
| `barrido_consolidado.csv` | una fila por configuración, ordenada por mIoU |
| `sweep_results*.csv` | salida cruda de cada partición del barrido |
