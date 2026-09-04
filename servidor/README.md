# Orquestación en el DGX

El entrenamiento corre en un DGX H200 de la universidad. Estos scripts existen
porque ese entorno tiene tres particularidades que rompen lo obvio:

1. **No hay PyTorch en el Python del sistema.** Es una máquina compartida y el
   `/usr/bin/python3` es el del sistema operativo; instalar ahí requeriría root y
   rompería el gestor de paquetes. El trabajo se hace en contenedores Docker, así
   que `entrenar.sh` detecta una imagen local que traiga torch y la usa.
2. **El `$HOME` dentro del contenedor es `/root` y es efímero.** Todo lo que se
   escriba ahí puede desaparecer. Por eso los scripts usan siempre rutas
   absolutas dentro del volumen montado, nunca `~`.
3. **Las GPUs están compartidas con otras personas.** Pedir una GPU ocupada no da
   un error claro, da un `OutOfMemory` que parece culpa del modelo. Por eso
   `entrenar_paralelo.sh` consulta la memoria libre antes de repartir el trabajo.

## Uso

```bash
bash servidor/lanzar.sh                              # una corrida
bash servidor/lanzar.sh --size 256 256 --base 64     # los flags van a train.py
bash servidor/lanzar.sh sweep --bases 32 64          # un barrido secuencial
bash servidor/lanzar.sh paralelo --bases 32 64 96    # repartido entre GPUs

bash servidor/estado.sh                              # progreso
bash servidor/recoger.sh                             # descarga resultados
```

`lanzar.sh` sube el código, sube el dataset la primera vez (como zip, porque un
archivo grande viaja mucho más rápido que 22 000 pequeños), y arranca el
entrenamiento desacoplado de la sesión SSH. Al volver el comando ya se puede
cerrar la terminal.

| script | dónde corre |
|---|---|
| `lanzar.sh` | local — sube y lanza |
| `entrenar.sh` | remoto — una corrida o un barrido |
| `entrenar_paralelo.sh` | remoto — reparte la rejilla entre las GPUs libres |
| `estado.sh` | local — progreso de lo que esté corriendo |
| `recoger.sh` | local — descarga modelos, métricas y logs |
| `diagnostico.sh` | local — inspecciona el entorno del servidor |

El servidor y el usuario están al principio de cada script (`SERVER=`), en una
sola línea, para poder apuntarlos a otra máquina.
