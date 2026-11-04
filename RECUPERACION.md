# Recuperación del entrenamiento Atari Go

El checkpoint 9400 conserva jugadas útiles, pero su valor devuelve ±1 según
el color incluso en posiciones donde negras pueden capturar inmediatamente.
La recuperación conserva el cuerpo y la política; reinicia la cabeza de valor
y su estado de AdamW, descarta el buffer sesgado y escribe en otra carpeta.
El original no se modifica. No garantiza recuperar fuerza de juego de inmediato.
Después del reinicio se entrenan todos los pesos; cuerpo y política se conservan
como punto de partida, no se congelan.

## Continuar el checkpoint recuperado incluido en el paquete

`atari_recovery_ready.zip` incluye el código y el checkpoint del piloto CPU en
`runs/atari9_recovery/latest.pt`. Extraer en una carpeta nueva, usar el Python
con PyTorch/CUDA del otro PC y ejecutar desde esa carpeta:

```text
python train_atari.py --resume runs/atari9_recovery/latest.pt --device cuda --threads 20 --workers 40 --games-per-iteration 80 --simulations 64 --inference-batch-size 40 --batch-wait-ms 1 --additional-iterations 200 --hours 0.5 --eval-every 20 --eval-games 20 --checkpoint-every 10
```

No añadir `--reset-value` ni `--clear-replay`: ya se aplicaron al crear este
checkpoint. Se heredan inicio mixto, learning rate 0.0001, 32 pasos de entrenamiento
y batch size 64. El primer piloto en GPU comprueba continuidad y rendimiento.
El modelo aún necesita entrenamiento; no es una versión final con fuerza validada.

## Recuperar desde el original en la 3070 (alternativa)

Copiar los archivos Python actualizados a la carpeta del proyecto del otro PC.
Mantener allí `runs/atari9_central3/latest.pt`. Desde esa carpeta ejecutar:

```text
python train_atari.py --resume runs/atari9_central3/latest.pt --output runs/atari9_recovery --reset-value --clear-replay --initial-position mixed --learning-rate 0.0001 --simulations 64 --device cuda --threads 20 --workers 40 --games-per-iteration 80 --inference-batch-size 40 --batch-wait-ms 1 --train-steps 32 --batch-size 64 --additional-iterations 200 --hours 0.5 --eval-every 20 --eval-games 20 --checkpoint-every 10
```

Es un piloto de hasta 200 iteraciones o media hora (límite blando al terminar
la iteración), no un entrenamiento de veinte horas. Usar un destino vacío.
Los 40 workers admiten como máximo 40 solicitudes simultáneas: un lote de 64
no aumentaba ese límite. Esta configuración no es un benchmark óptimo de GPU.

## Continuar después del primer arranque

```text
python train_atari.py --resume runs/atari9_recovery/latest.pt --device cuda --threads 20 --workers 40 --inference-batch-size 40 --batch-wait-ms 1 --additional-iterations 200 --hours 0.5 --checkpoint-every 10
```

No repetir `--reset-value` ni `--clear-replay`: perderían lo recuperado.
Los parámetros de aprendizaje se heredan si no se especifican; los de ejecución
(workers, threads, dispositivo, frecuencia de checkpoint) se indican cada vez.
`--output` distinto crea una rama nueva y conserva número de iteración y tiempo
acumulados. `recovery.json` registra origen y operaciones de recuperación.
Los `model_XXXXX.pt` son modelos para jugar/evaluar; reanudar requiere `latest.pt`.

## Cambios de aprendizaje

- BCE calculada sobre `2 * valor_crudo`, equivalente a la probabilidad asociada
  a `tanh(valor_crudo)`. Una predicción equivocada saturada recibe gradiente.
- Objetivos suavizados a probabilidades 0.01/0.99 para limitar confianza excesiva.
- Intercambio aleatorio de colores absolutos al aumentar datos. Los planos de
  piedras son relativos; solo cambia el plano del color al turno. Atari Go no
  tiene komi, por lo que política y resultado siguen siendo válidos.
- Inicio `mixed` (tableros >=5): aproximadamente 25% central con apertura
  aleatoria, 25% vacío con apertura aleatoria, 25% captura y 25% defensa.
  Los prefijos aleatorios excluyen movimientos terminales y no se etiquetan
  como decisiones MCTS. Las posiciones tácticas son ejercicios sintéticos
  legales, rotados/reflejados y con ambos colores al turno.
- Se detiene antes de actualizar pesos si aparecen pérdidas o gradientes no finitos.
- La búsqueda comprueba capturas inmediatas en la raíz. Si hay una amenaza
  contra un grupo propio, descarta movimientos que permiten una captura rival
  inmediata cuando existe alguna alternativa segura. Si todos pierden, conserva
  la búsqueda. Esta protección usa reglas, no depende del valor aprendido, y
  también mejora el autojuego que produce las etiquetas de política.

## Qué comprobar

`metrics.jsonl` registra:

| Campo | Interpretación |
|---|---|
| `value_loss` | Nueva BCE; **no comparar directamente** con la antigua MSE |
| `value_mse` | Error cuadrático sobre resultados ±1, comparable en definición pero no en distribución |
| `value_saturation` | Fracción del lote con `abs(valor) >= 0.999`, antes de cada actualización |
| `selfplay_white_win_rate` | Fracción de partidas ganadas por blancas; no exigir exactamente 50% |
| `mean_game_moves` | Jugadas aprendidas desde el inicio seleccionado; excluye el prefijo aleatorio |
| `tactical_capture_rate` | Frecuencia de elegir la captura inmediata en 8 ejercicios fijos |
| `tactical_defense_rate` | Frecuencia de elegir la única extensión segura en 8 ejercicios fijos |
| `tactical_*_guarded_rate` | Las mismas pruebas con protección táctica activada |
| `tactical_*_value_black/white` | Valor de esos ejercicios desde el jugador al turno |
| `tactical_*_color_gap` | Diferencia absoluta al intercambiar colores en la misma situación relativa |

Los empates entre máximos MCTS aportan su probabilidad exacta a la tasa táctica.
Las tasas sin `guarded` desactivan esa protección para medir red + MCTS sin
ocultar errores aprendidos. Las partidas normales usan la protección.
Estas pruebas son controles básicos, similares a ejercicios del entrenamiento;
no son una medición independiente de fuerza ni una prueba de generalización.
Una saturación baja justo después del reinicio es esperable y **no basta**:
deben mejorar las decisiones y la valoración de capturas. Más de 90% de
saturación emite una advertencia. El 100% contra aleatorio tampoco basta.
Las partidas desde ejercicios de captura pueden durar una sola jugada por
diseño. En el modo mixto, exigir partidas largas no es un criterio de calidad.

```text
python evaluate_atari.py runs/atari9_recovery/latest.pt --device cuda --simulations 128 --tactics-only
python evaluate_atari.py runs/atari9_recovery/latest.pt --opponent runs/atari9_central3/latest.pt --initial-position central --device cuda --simulations 128 --games 40
python evaluate_atari.py runs/atari9_recovery/latest.pt --opponent runs/atari9_central3/latest.pt --initial-position mixed --device cuda --simulations 128 --games 40
python diagnose_atari.py runs/atari9_recovery/latest.pt --device cuda --output diagnostico_recuperado.json
```

Interpretar resultados por color y por posición inicial. El inicio central
puede producir victorias de blancas con ambos modelos y ocultar diferencias.
Con `mixed`, cada pareja de partidas comparte posición inicial e intercambia
el color de los modelos. Las aperturas tienen semilla fija. Aun así, una muestra
pequeña no constituye una clasificación de fuerza.
`play.py` y `watch_atari.py` siguen permitiendo elegir inicio central o vacío.

`diagnose_atari.py` genera posiciones mediante juego aleatorio desde tableros
centrales y vacíos, independientes de los pesos. Busca 40 oportunidades de
victoria inmediata, comprueba su valoración por color y la probabilidad de
elegir una captura con 32/128 simulaciones. También mide saturación y diferencia
de valor al intercambiar colores. Desactiva la protección táctica para no dar
100% artificialmente por la comprobación de reglas. Comparar checkpoints con
la misma semilla.
Estas posiciones pueden estar correlacionadas dentro de una partida; la tasa
es un diagnóstico, no un intervalo estadístico de fuerza general.

## Validación local

```text
python -m unittest discover -v
```

Incluye reglas, autojuego paralelo y secuencial mixto, gradiente de una predicción
equivocada saturada, reinicio selectivo, argumentos de reanudación y conservación
del checkpoint fuente. La validación local se realiza en CPU; CUDA debe
comprobarse en el piloto de la 3070.
