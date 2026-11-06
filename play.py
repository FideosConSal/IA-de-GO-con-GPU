"""Interfaz de consola para jugar contra un agente aleatorio o entrenado."""

from __future__ import annotations

import argparse
from typing import Optional

from go_env import (
    BLACK,
    WHITE,
    GoEnv,
    IllegalMoveError,
    PLAYER_NAMES,
    RandomAgent,
    format_move,
    parse_human_move,
)


def _ask_choice(prompt: str, choices: dict[str, str], default: str) -> str:
    """Solicita una opción hasta recibir una entrada válida."""
    while True:
        raw = input(prompt).strip().lower()
        if not raw:
            return default
        if raw in choices:
            return choices[raw]
        print(f"Opción inválida. Valores permitidos: {', '.join(choices)}.")


def _announce_result(env: GoEnv, result) -> None:
    """Imprime información de una jugada y, si corresponde, el resultado."""
    player_name = PLAYER_NAMES[result.player].capitalize()
    if result.was_pass:
        print(f"{player_name} pasa.")
    else:
        print(f"{player_name} juega {format_move(result.move, env.size)}.")

    if not result.was_pass:
        if result.captured_count:
            print(
                f"Captura: {player_name} retiró {result.captured_count} "
                f"piedra(s)."
            )
        else:
            print("Capturas: ninguna.")

    print()
    print(env.render())
    print()

    if not result.done:
        return

    if result.winner is None:
        print("La partida termina en empate.")
    else:
        print(f"Ganador: {PLAYER_NAMES[result.winner].capitalize()}.")

    if result.reason == "first_capture":
        print("Motivo: primera captura en Atari Go.")
    elif result.reason == "no_legal_moves":
        print("Motivo: el siguiente jugador no tiene movimientos legales.")
    elif result.reason in ("double_pass", "move_limit") and result.score is not None:
        if result.reason == 'move_limit':
            print('Motivo: límite de jugadas; adjudicación por área del tablero actual.')
        score = result.score
        print(
            "Conteo de área: "
            f"negro {score.black_score:.1f}, "
            f"blanco {score.white_score:.1f} "
            f"(komi {score.komi:.1f})."
        )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Juega Go estándar o Atari Go contra un agente aleatorio o entrenado."
    )
    parser.add_argument("--size", type=int, help="Tamaño; con modelo se obtiene del checkpoint.")
    parser.add_argument("--checkpoint", help="Ruta a latest.pt o model_XXXXX.pt; la variante se obtiene del modelo.")
    parser.add_argument("--initial-position", choices=("central", "empty"),
                        help="Atari Go usa central por defecto; Go estándar usa empty.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--simulations", type=int, default=32, help="Simulaciones MCTS por jugada.")
    parser.add_argument(
        "--mode",
        choices=("standard", "atari"),
        help="Variante. Si se omite, se pregunta al comenzar.",
    )
    parser.add_argument(
        "--human-color",
        choices=("black", "white"),
        help="Color humano. Si se omite, se pregunta al comenzar.",
    )
    parser.add_argument("--komi", type=float)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    model = None
    if args.simulations < 1:
        parser.error("--simulations debe ser positivo.")
    if args.checkpoint:
        try:
            import numpy as np
            import torch
            from az_atari import search
            from evaluate_atari import load_model
        except ImportError:
            parser.error("Instala las dependencias de requirements-train.txt para usar modelos.")
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA no disponible. Usa --device cpu o instala PyTorch con CUDA.")
        torch.set_num_threads(2)
        try:
            model, config = load_model(args.checkpoint, args.device)
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            parser.error(f"No se pudo cargar el modelo: {exc}")
        if args.size is not None and args.size != config['size']:
            parser.error("--size debe coincidir con el tamaño del checkpoint.")
        args.size = config['size']
        checkpoint_mode = config.get('game_mode', 'atari')
        if args.mode is not None and args.mode != checkpoint_mode:
            parser.error('--mode debe coincidir con el checkpoint.')
        args.mode = checkpoint_mode
        if checkpoint_mode == 'standard':
            if args.komi is not None and args.komi != config['komi']:
                parser.error('--komi debe coincidir con el checkpoint.')
            args.komi = config['komi']
        rng = np.random.default_rng(args.seed)

    mode = args.mode or _ask_choice(
        "Modo [standard/atari] (standard): ",
        {"standard": "standard", "s": "standard", "atari": "atari", "a": "atari"},
        "standard",
    )
    human_color_name = args.human_color or _ask_choice(
        "Color humano [black/white] (black): ",
        {"black": "black", "b": "black", "white": "white", "w": "white"},
        "black",
    )
    human_player = BLACK if human_color_name == "black" else WHITE

    try:
        env = GoEnv(size=args.size if args.size is not None else 9, game_mode=mode,
                    komi=args.komi if args.komi is not None else 7.5, initial_position=args.initial_position,
                    max_moves=config.get('max_moves') if model is not None and mode == 'standard' else None)
    except ValueError as exc:
        parser.error(str(exc))
    machine = RandomAgent(seed=args.seed)

    print()
    print(f"Modo: {mode}; tablero: {env.size}x{env.size}.")
    print(f"Posición inicial: {env.initial_position}.")
    if model is not None:
        print(f"Modelo: {args.checkpoint}; MCTS: {args.simulations}; dispositivo: {args.device}.")
    print("Coordenadas: D4 (fila desde abajo, sin columna I) o 3,4")
    print("(fila desde arriba, columna desde la izquierda; ambas 1-based).")
    if mode == "standard":
        print("Escriba 'pass' para pasar.")
        if env.max_moves:
            print(f'Límite del modelo: {env.max_moves} jugadas; al alcanzarlo se cuenta el área.')
    print("Escriba 'quit' para abandonar la interfaz.")
    print()
    print(env.render())
    print()

    while not env.done:
        legal_moves = env.legal_moves()
        current = env.current_player

        if current == human_player:
            while True:
                try:
                    raw = input(f"Turno de {PLAYER_NAMES[current]}> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nPartida interrumpida.")
                    return
                if raw.lower() in {"quit", "exit", "salir"}:
                    print("Partida interrumpida.")
                    return
                try:
                    move = parse_human_move(
                        raw,
                        size=env.size,
                        allow_pass=(env.game_mode == "standard"),
                    )
                except ValueError as exc:
                    print(f"Entrada inválida: {exc}")
                    continue

                if move not in legal_moves:
                    print(
                        "Movimiento ilegal: la intersección puede estar ocupada, "
                        "ser suicidio o violar ko."
                    )
                    continue
                break
        else:
            print(f"La máquina ({PLAYER_NAMES[current]}) está jugando...", flush=True)
            if model is None:
                move = machine.select_move(env, legal_moves)
            else:
                try:
                    policy = search(env, model, args.device, rng, args.simulations, noise=False)
                except KeyboardInterrupt:
                    print("\nPartida interrumpida.")
                    return
                move = int(rng.choice(np.flatnonzero(policy == policy.max())))

        try:
            result = env.play_move(move)
        except IllegalMoveError as exc:
            # No debería ocurrir después de usar legal_moves, pero deja la CLI
            # protegida ante modificaciones futuras del entorno.
            print(f"No se pudo aplicar la jugada: {exc}")
            continue

        _announce_result(env, result)


if __name__ == "__main__":
    main()
