# Entrenamiento de Go estándar

`train_go.py` entrena una red de política/valor mediante MCTS y autojuego.
Una captura continúa la partida. El tablero comienza vacío, hay una acción de
pase y dos pases consecutivos terminan con conteo de área y komi.

## Iniciar en la RTX 3070

Después de actualizar el código con tu flujo de commit/push/pull, ejecutar
desde la carpeta del repositorio cuando quieras iniciar este entrenamiento:

```text
python train_go.py --size 9 --komi 7.5 --output runs/go9 --device cuda --threads 20 --workers 40 --games-per-iteration 80 --simulations 64 --inference-batch-size 40 --batch-wait-ms 1 --learning-rate 0.0001 --train-steps 32 --batch-size 64 --additional-iterations 200 --hours 0.5 --eval-every 10 --eval-games 20 --checkpoint-every 5
```

Este primer comando es un piloto de hasta 200 iteraciones o media hora, con
límite de tiempo blando al terminar una iteración. No es una configuración
de rendimiento optimizada. Las partidas completas pueden ser mucho más largas
que en Atari, y los workers se limitan por el número de partidas de la ronda.

El modelo de Go estándar se inicia desde cero. `--resume` no permite cargar
modelos Atari: tienen otro espacio de acciones y otro significado del resultado.
`train_atari.py` conserva su comportamiento y sus checkpoints son compatibles.

## Reanudar

```text
python train_go.py --resume runs/go9/latest.pt --device cuda --threads 20 --workers 40 --inference-batch-size 40 --batch-wait-ms 1 --additional-iterations 100000000 --hours 20 --checkpoint-every 5
```

Se recuperan arquitectura, komi, límite de jugadas y parámetros del experimento.
Los argumentos de aprendizaje explícitos sustituyen los guardados, excepto
tamaño, arquitectura, komi y límite, que deben coincidir: cambiar las reglas
invalidaría las etiquetas del replay. `--additional-iterations` cuenta desde el
checkpoint; `--iterations` es el total absoluto. No se combinan las dos opciones.
Los parámetros de ejecución (workers, threads, dispositivo y frecuencia de
checkpoint) se indican en cada invocación. `Ctrl+C` conserva el último checkpoint
completo, sin guardar una actualización parcial.

## Evaluar, observar y jugar

```text
python evaluate_go.py runs/go9/latest.pt --device cuda --games 40 --simulations 128
python evaluate_go.py runs/go9/latest.pt --opponent runs/go9/model_00010.pt --device cuda --games 40 --simulations 128
python watch_go.py runs/go9/latest.pt --device cuda --games 2 --simulations 128 --delay 0.2
python play.py --checkpoint runs/go9/latest.pt --human-color black --device cuda --simulations 128
```

Los enfrentamientos alternan colores. Para comparar modelos se exige igual
tablero, variante, komi y límite. La evaluación devuelve victorias por color,
empates, diferencia media de puntaje negro menos blanco y fracción de cierres
por límite. Ganar contra aleatorio no demuestra fuerza general.

## Reglas implementadas y límite de entrenamiento

- Ko simple y suicidio prohibido, según `go_env.py`. **No hay superko**.
- Conteo de área: piedras en el tablero más regiones vacías rodeadas por un
  único color; el komi se suma a blancas. Los empates producen objetivo de valor 0.
- No se retiran piedras muertas por acuerdo al final: deben capturarse durante
  la partida. No equivale a un árbitro completo de todas las reglas de torneo.
- Para acotar ciclos de autojuego, `--max-moves` vale por defecto `4 * size²`
  (324 en 9×9). Al alcanzar el límite se adjudica por el área del tablero actual,
  con razón `move_limit`, que puede puntuar posiciones aún sin resolver. El MCTS
  usa exactamente el mismo límite. `--max-moves 0` selecciona el valor automático
  al iniciar; al reanudar omitirlo o indicar el valor efectivo guardado.
- Los pases son decisiones aprendidas desde el primer turno; no se fuerzan
  pases ni se eliminan jugadas de captura/defensa por reglas de Atari.

## Métricas para revisar

Además de pérdidas, saturación y victorias de blancas, se registran:

| Métrica | Significado |
|---|---|
| `double_pass_rate` | Fracción de partidas terminadas por dos pases |
| `move_limit_rate` | Fracción adjudicada artificialmente por límite |
| `mean_passes` | Cantidad media de pases, incluidos los no consecutivos |
| `mean_game_moves` | Longitud media, contando pases |
| `mean_score_margin` | Diferencia absoluta media del puntaje final |
| `draw_rate` | Fracción de empates |

Una tasa alta de `move_limit_rate` indica que la evaluación depende mucho de
adjudicaciones; no interpretar esas victorias como dominio del final de partida.
Partidas de dos pases desde el comienzo también pueden indicar un comportamiento
degenerado: revisar longitud, puntaje y partidas observadas, no solo las pérdidas.

La red recibe nueve planos: los seis anteriores, pases consecutivos, komi relativo
al jugador y fracción de jugadas restantes. La política añade un logit de pase
al final (índice `size²`). Las rotaciones y reflexiones conservan ese componente.
No se intercambian colores durante el aumento de datos estándar: con komi fijo
esa transformación no preservaría las etiquetas. La pérdida de valor usa BCE
sobre logits con suavizado, como el entrenador Atari recuperado.

## Validación local

```text
python -m unittest discover -v
```

Se verificaron pases, capturas no terminales, empates, límite y copia del estado,
simetrías con komi, coincidencia de autojuego paralelo/secuencial, entrenamiento
del pase, reanudación, rechazo de variantes incompatibles y regresiones de Atari.
La prueba funcional CPU en `runs/go_standard_smoke` ejecutó 3 iteraciones y
reanudó una cuarta: 16 partidas en total, con evaluación y guardado.
Es una prueba de funcionamiento, no un
modelo con fuerza de juego validada. CUDA debe comprobarse en el otro PC.

Todo el código y esta guía están dentro del repositorio. `runs/` sigue ignorado
por Git: los pesos generados en cada PC se gestionan aparte.
