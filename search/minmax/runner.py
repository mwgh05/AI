from importlib import import_module
from time import perf_counter
from typing import Dict, List

from model import apply_move, create_initial_state, is_terminal, move_to_text, render_board
from search_ai import minmax


def play_game_with_minmax(n: int = 3, m: int = 3, depth: int = 5) -> Dict[str, object]:
    state = create_initial_state(n, m)
    history: List[Dict[str, object]] = []
    stats = {"nodes": 0}
    start = perf_counter()
    move_count = 0

    while not is_terminal(state):
        current_player = state.player
        value, best = minmax(
            state=state,
            depth=depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing_player=current_player,
            stats=stats,
        )

        if best is None:
            break

        before_scores = state.scores
        state = apply_move(state, best)
        after_scores = state.scores

        history.append(
            {
                "turn": move_count + 1,
                "player": current_player,
                "move": best,
                "move_text": move_to_text(best),
                "minmax_value": value,
                "score_before": before_scores,
                "score_after": after_scores,
                "next_player": state.player,
                "board": render_board(state),
            }
        )
        move_count += 1

    elapsed = perf_counter() - start

    return {
        "final_state": state,
        "history": history,
        "moves": move_count,
        "elapsed_seconds": elapsed,
        "nodes_explored": stats["nodes"],
    }


def print_game_result(result: Dict[str, object]) -> None:
    final_state = result["final_state"]
    history = result["history"]

    print("=== PATH (estado inicial -> estado objetivo) ===")
    print("Tablero inicial:")
    if history:
        n = final_state.n
        m = final_state.m
        print(render_board(create_initial_state(n, m)))
    else:
        print(render_board(final_state))

    for step in history:
        print(
            f"Turno {step['turn']}: J{step['player']} juega {step['move_text']} | "
            f"score {step['score_before']} -> {step['score_after']} | "
            f"siguiente J{step['next_player']}"
        )
        print(step["board"])
        print()

    print("\n=== FINAL BOARD ===")
    print(render_board(final_state))

    print("\n=== METRICS ===")
    print(f"Movimientos totales: {result['moves']}")
    print(f"Tiempo de ejecucion: {result['elapsed_seconds']:.6f} s")
    print(f"Nodos explorados: {result['nodes_explored']}")
    print(f"Puntaje final: {final_state.scores}")


def run_benchmark(n: int, m: int, min_depth: int, max_depth: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    print("=== BENCHMARK ===")
    for depth in range(min_depth, max_depth + 1):
        result = play_game_with_minmax(n=n, m=m, depth=depth)
        row = {
            "depth": float(depth),
            "time": float(result["elapsed_seconds"]),
            "nodes": float(result["nodes_explored"]),
            "moves": float(result["moves"]),
        }
        rows.append(row)
        print(
            f"depth={depth} | time={row['time']:.6f}s | "
            f"nodes={int(row['nodes'])} | moves={int(row['moves'])}"
        )

    try:
        plt = import_module("matplotlib.pyplot")

        depths = [int(r["depth"]) for r in rows]
        times = [r["time"] for r in rows]
        nodes = [r["nodes"] for r in rows]

        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        color1 = "tab:blue"
        ax1.set_xlabel("Depth")
        ax1.set_ylabel("Time (s)", color=color1)
        ax1.plot(depths, times, marker="o", color=color1, label="Time")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("Nodes", color=color2)
        ax2.plot(depths, nodes, marker="s", color=color2, label="Nodes")
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title(f"Minmax Performance ({n}x{m} points)")
        fig.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"No se pudo mostrar grafica: {exc}")

    return rows
