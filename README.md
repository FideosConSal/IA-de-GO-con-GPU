# Go RL Baseline

Para entrenar **Go estándar** (pases, conteo de área y komi), ver
[GO_ESTANDAR.md](GO_ESTANDAR.md). Se inicia con `python train_go.py`.

Para recuperar el entrenamiento 9×9 saturado y preparar la RTX 3070,
ver [RECUPERACION.md](RECUPERACION.md).

## Entrenamiento inicial: Atari Go 5×5

Se agregó una implementación pequeña estilo AlphaZero basada en PyTorch:
red residual de política/valor, búsqueda PUCT/MCTS, autojuego con ruido
Dirichlet, aumentos por las ocho simetrías del tablero y AdamW. No requiere
Gymnasium ni datos humanos. Es una primera versión de una sola máquina;
con `--workers` agrupa consultas de varios procesos MCTS en una sola GPU.
No se garantiza una fuerza determinada por ejecutar unas horas.

### Instalación (PowerShell)

Usar Python 3.10 o superior y un entorno virtual. Para validar en CPU:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe -m unittest -v
```

En el equipo con RTX 3070, instalar primero la distribución de PyTorch con
CUDA siguiendo el selector oficial <https://pytorch.org/get-started/locally/>.
Usar el Python de `.venv` en lugar de `pip` global. Si ya hay una distribución
CPU, reemplazarla con `--upgrade --force-reinstall` usando el índice CUDA
indicado por el selector. Después instalar `requirements-train.txt`.
Comprobar la detección antes de un entrenamiento largo:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### GPU y generación paralela en un PC de 20 hilos

```powershell
.\.venv\Scripts\python.exe train_atari.py --device cuda --workers 18 --threads 2 --games-per-iteration 72 --inference-batch-size 32 --batch-wait-ms 2 --size 5 --iterations 10000 --hours 2 --output runs/atari5_gpu_parallel
```

`--workers` controla procesos de CPU con partidas independientes; cada uno usa
un hilo para las bibliotecas numéricas. `--threads` controla los hilos PyTorch
del proceso principal, no la cantidad de partidas. Se sugieren 18 procesos
para dejar capacidad al proceso principal y al sistema; se puede probar 20.
No garantiza ocupar todos los hilos permanentemente ni superar al modo serial
en 5×5: medir `self_play_seconds` e `iteration_seconds` en el equipo real.

Los workers calculan reglas/MCTS, y solo el proceso principal carga el modelo
en GPU. Las consultas se agrupan hasta `--inference-batch-size` o durante
`--batch-wait-ms` milisegundos. Hay como máximo una consulta pendiente por
worker; un lote nunca supera los workers activos. Los procesos se reutilizan
entre iteraciones; el arranque inicial en Windows puede tardar y consumir RAM.
Los pesos permanecen fijos durante cada ronda de autojuego, y se actualizan
después de recibir todas sus partidas. La evaluación periódica sigue siendo
secuencial. La GPU no estará necesariamente al 100 % durante toda la ejecución.

`--games-per-iteration` debe ser al menos la cantidad de workers para utilizarlos
todos; con menos partidas se limita la cantidad de procesos automáticamente.
Cambiar de 8 a 72 partidas cambia la cantidad de datos por actualización: es un
punto de partida para medir rendimiento, no una configuración de aprendizaje
validada. Los parámetros de ejecución se registran en `runtime.json` y métricas,
incluyendo el tamaño medio y máximo de los lotes de inferencia.

Se puede reanudar con `--resume .../latest.pt --device cuda --workers 18`.
Los parámetros de ejecución se eligen en cada invocación; los del experimento
se recuperan del checkpoint salvo los que se indiquen explícitamente en el comando.
Por ejemplo, `--games-per-iteration 80` sí reemplaza el valor guardado.
No se puede cambiar tamaño ni arquitectura al reanudar. `--workers 1` conserva
el camino secuencial. Las semillas se asignan por partida y los resultados se
almacenan en orden estable, aunque cambios de dispositivo o agrupación pueden
producir diferencias numéricas. Ctrl+C cierra los workers y conserva el último
checkpoint completo.

### Prueba funcional corta

Los entrenamientos nuevos de Atari Go usan cuatro piedras centrales. En 5×5:
negras en D4 y C3, blancas en C4 y D3; negras al turno. En 9×9 son
negras en F6 y E5, blancas en E6 y F5 (blancas en la diagonal superior
izquierda/inferior derecha). Son piedras de
preparación, no jugadas, y no cuentan como capturas. Para comenzar vacío usar
`--initial-position empty`. La elección queda guardada en la configuración.
Al reanudar un checkpoint anterior a esta opción, se conserva el inicio vacío;
para cambiarlo indicar explícitamente `--initial-position central` o `mixed`.
Go estándar continúa comenzando vacío.

`play.py` y `watch_atari.py` usan la posición central por defecto incluso al
cargar modelos antiguos; aceptan `--initial-position empty` para jugar bajo
las condiciones anteriores. `evaluate_atari.py` usa la posición guardada en el
modelo (vacía en modelos antiguos); `--initial-position central` permite
evaluar explícitamente desde la nueva posición. Ambos rivales siempre parten
del mismo tablero en una evaluación.

```powershell
.\.venv\Scripts\python.exe train_atari.py --device cpu --channels 8 --blocks 1 --simulations 2 --games-per-iteration 2 --train-steps 2 --batch-size 8 --eval-every 1 --eval-games 2 --iterations 2 --output runs/smoke
```

Esta configuración verifica el circuito; no sirve para medir fuerza.
Cada ejecución nueva necesita una carpeta vacía. Para la primera prueba real
en la 3070 (4 bloques, 64 canales, 32 simulaciones por defecto):

```powershell
.\.venv\Scripts\python.exe train_atari.py --device cuda --iterations 10000 --hours 2 --output runs/atari5_seed7
```

El límite de tiempo es aproximado: se comprueba entre iteraciones. Las métricas
indican partidas/hora durante autojuego, segundos por iteración, pérdidas y
victorias frente al agente aleatorio cada cinco iteraciones (20 partidas).
La primera versión puede estar limitada por CPU; una GPU poco ocupada no
implica que el programa esté detenido.

### Jugar como humano contra un modelo

```powershell
.\.venv\Scripts\python.exe play.py --checkpoint runs/atari5_prueba/latest.pt --human-color black
```

También acepta checkpoints históricos `model_XXXXX.pt`. El tamaño del tablero
se obtiene del modelo y el modo se fija en Atari Go. Para jugar con blancas,
usar `--human-color white`; el modelo hace la primera jugada. Escribir `C3`
y Enter para colocar una piedra, o `quit` para salir. Gana la primera captura.
`--simulations 64` aumenta la búsqueda por turno; `--device cuda` permite usar
la GPU cuando PyTorch tiene CUDA disponible. Por defecto usa CPU y 32 simulaciones.
Sin `--checkpoint`, se mantiene el juego contra el agente aleatorio.

### Ver partidas bot contra bot

Ver el modelo contra sí mismo en la consola, con un segundo entre movimientos:

```powershell
.\.venv\Scripts\python.exe watch_atari.py runs/atari5_prueba/latest.pt
```

Comparar dos modelos, alternando colores en dos partidas:

```powershell
.\.venv\Scripts\python.exe watch_atari.py runs/atari5_prueba/latest.pt --opponent runs/atari5_prueba/model_00005.pt --games 2 --simulations 64
```

`--random` enfrenta al modelo contra el bot aleatorio. `--step` permite avanzar
con Enter (o salir con `quit`); `--delay 0.5` cambia la pausa automática y
`--delay 0` la elimina. Ctrl+C detiene la visualización. Se muestran el tablero,
las jugadas, capturas, ganador y marcador A/B. No modifica los modelos.

### Guardar, reanudar y evaluar

`latest.pt` contiene red, optimizador, buffer y estados aleatorios. Se reemplaza
de forma atómica cada 10 iteraciones por defecto y al terminar normalmente
por límite de tiempo o iteraciones. `--checkpoint-every 50` cambia la frecuencia
a 50; se puede indicar también al reanudar. Se crea un checkpoint inicial en
la iteración 0. `Ctrl+C` conserva el último checkpoint guardado y descarta todo
el trabajo posterior a ese guardado, incluso iteraciones completas sin guardar.
Los pesos históricos `model_00005.pt`, etc., sirven para evaluación y se guardan
cada intervalo de evaluación. Solo `latest.pt` permite reanudar entrenamiento.

```powershell
.\.venv\Scripts\python.exe train_atari.py --resume runs/atari5_seed7/latest.pt --device cuda --iterations 10000 --hours 2
.\.venv\Scripts\python.exe evaluate_atari.py runs/atari5_seed7/latest.pt --device cuda --games 100
.\.venv\Scripts\python.exe evaluate_atari.py runs/atari5_seed7/latest.pt --opponent runs/atari5_seed7/model_00005.pt --device cuda --games 100
```

Al reanudar se recupera la configuración del checkpoint; `--iterations` es el
total acumulado; `--additional-iterations N` permite pedir N iteraciones más
sin calcular ese total. No se combinan ambas opciones. `--hours` se aplica a esa invocación. Se puede cambiar el
dispositivo. `config.json` registra parámetros y `metrics.jsonl` los resultados
por iteración. Un fallo entre guardar y escribir métricas puede dejar la última
fila ausente; el checkpoint determina la iteración real al reanudar.

Las evaluaciones alternan colores, no usan ruido y dan el mismo número de
simulaciones a ambos modelos. El intervalo Wilson informado describe las
partidas evaluadas, no la variabilidad entre entrenamientos independientes.
Las evaluaciones periódicas usan una semilla fija; usar semillas nuevas para
la evaluación final. Ganar al azar no demuestra fuerza contra otros agentes.

El entrenador solo admite Atari Go (sin pase). Al terminar la primera captura,
cada posición recibe +1 o -1 desde la perspectiva del jugador que debía mover.
Como no hay capturas anteriores al final, las partidas duran a lo sumo N²
jugadas. La red acepta otros tamaños con `--size`, pero todavía no implementamos
la transferencia a Go estándar, su acción de pase ni entrenamiento distribuido.

Archivos nuevos: `az_atari.py` (red y aprendizaje), `train_atari.py` (CLI y
checkpoints), `evaluate_atari.py` (partidas comparativas), `test_training.py`.

Base en Python/NumPy para investigar aprendizaje por refuerzo en Go sobre hardware convencional. Incluye reglas de Go estándar, Atari Go, un agente aleatorio, una interfaz de consola y pruebas unitarias.

## Archivos

- `go_env.py`: reglas, estado, codificación para ML y `RandomAgent`.
- `play.py`: humano contra agente aleatorio.
- `test_go_env.py`: pruebas de captura, suicidio, ko, Atari Go y utilidades.

## Instalación y ejecución

```bash
python -m pip install numpy
python -m unittest -v test_go_env.py
python play.py --size 9 --mode atari --human-color black --seed 7
python play.py --size 9 --mode standard --human-color white --komi 7.5
```

## API mínima

```python
from go_env import GoEnv, RandomAgent

env = GoEnv(size=9, game_mode="atari")
agent = RandomAgent(seed=0)

while not env.done:
    legal = env.legal_moves()
    move = agent.select_move(env, legal)
    observation, reward, done, info = env.step(move)

print(env.winner, info)
```

Las acciones de tablero son `fila * N + columna`. El pass es `N*N` y solo existe en modo estándar. `action_mask()` devuelve una máscara directamente utilizable para anular logits o Q-values de acciones ilegales.

## Representación para redes

`encode_state()` devuelve `(N, N, 3)` con planos negro, blanco y vacío. Para entrenamiento real conviene usar:

```python
x = env.encode_state(
    channels_last=False,
    include_to_play=True,
    include_history=True,
)
```

Esto produce seis planos: negro actual, blanco actual, vacío, jugador al turno, negro anterior y blanco anterior. Los dos planos históricos permiten representar la información del ko simple; sin historial, el tablero por sí solo no es completamente markoviano.

## Integración con Gym/Gymnasium

La clase ya expone `reset()`, `step()`, `action_size`, `legal_moves()` y `action_mask()`. Un adaptador puede declarar un espacio `Discrete(env.action_size)` y un espacio `Box(0, 1, shape=(N, N, C))`. En un entorno de dos jugadores hay que fijar una convención: controlar un solo color y usar al rival dentro del wrapper, o producir transiciones de autojuego con el color del actor asociado a cada estado.

Esquema de un wrapper opcional:

```python
# Requiere gymnasium, no forma parte de las dependencias del núcleo.
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GymGoWrapper(gym.Env):
    def __init__(self, size=9, game_mode="standard"):
        self.game = GoEnv(size=size, game_mode=game_mode)
        channels = 6
        self.action_space = spaces.Discrete(self.game.action_size)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(channels, size, size),
            dtype=np.float32,
        )

    def _obs(self):
        return self.game.encode_state(
            channels_last=False,
            include_to_play=True,
            include_history=True,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._obs(), {"action_mask": self.game.action_mask()}

    def step(self, action):
        _, reward, terminated, info = self.game.step(action)
        info["action_mask"] = self.game.action_mask()
        return self._obs(), reward, terminated, False, info
```

## Extensión hacia un agente de aprendizaje

Para DQN, una CNN puede producir `action_size` Q-values; antes de elegir la acción se asigna `-inf` a acciones donde `action_mask()` sea falsa. DQN es posible, pero Go es adversarial, tiene recompensas escasas y un espacio combinatorio grande.

Para un AlphaZero simplificado, la red tendría dos cabezas: una política de `action_size` logits y un valor escalar en `[-1, 1]`. Cada nodo de MCTS usa `env.copy()` para simular acciones. Durante autojuego se guardan `(estado, política_MCTS, jugador_al_turno)` y, al terminar, se convierte `env.winner` en un objetivo de valor desde la perspectiva del jugador guardado.

## Decisiones y límites

- Los grupos y libertades se calculan por búsqueda en profundidad sobre vecinos ortogonales. Es deliberadamente simple y verificable; no está optimizado con cadenas incrementales o Zobrist hashing.
- El ko implementado es ko simple, exactamente una repetición inmediata. No hay superko.
- El conteo usa área determinista: piedras más regiones vacías rodeadas por un solo color, con komi configurable. No existe una fase humana para marcar piedras muertas; en autojuego deben capturarse antes de los dos pases.
- La generación legal simula cada intersección. Es suficiente como referencia en 9x9 y funcional en 19x19, aunque MCTS intensivo requerirá cachés, hashing y actualizaciones incrementales.
