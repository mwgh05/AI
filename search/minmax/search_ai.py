from typing import Dict, Optional, Tuple

from model import (
    GameState,
    Move,
    apply_move,
    box_index,
    box_sides_count,
    get_possible_moves,
    is_terminal,
)


def evaluate_state(state: GameState, maximizing_player: int) -> float:
    opponent = 1 - maximizing_player
    score_diff = state.scores[maximizing_player] - state.scores[opponent]

    my_three = 0
    opp_three = 0
    my_two = 0
    opp_two = 0

    for br in range(state.n - 1):
        for bc in range(state.m - 1):
            idx = box_index(state, br, bc)
            if state.box_owner[idx] != -1:
                continue

            sides = box_sides_count(state, br, bc)
            if sides == 3:
                if state.player == maximizing_player:
                    my_three += 1
                else:
                    opp_three += 1
            elif sides == 2:
                if state.player == maximizing_player:
                    my_two += 1
                else:
                    opp_two += 1

    return 10 * score_diff + 4 * (my_three - opp_three) + 1 * (my_two - opp_two)


def minmax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: int,
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[float, Optional[Move]]:
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + 1

    if depth == 0 or is_terminal(state):
        return evaluate_state(state, maximizing_player), None

    moves = get_possible_moves(state)
    if not moves:
        return evaluate_state(state, maximizing_player), None

    def move_priority(mv: Move) -> int:
        child = apply_move(state, mv)
        gained = child.scores[state.player] - state.scores[state.player]
        return gained

    moves.sort(key=move_priority, reverse=True)

    if state.player == maximizing_player:
        best_val = float("-inf")
        best_move: Optional[Move] = None

        for mv in moves:
            child = apply_move(state, mv)
            val, _ = minmax(child, depth - 1, alpha, beta, maximizing_player, stats)

            if val > best_val:
                best_val = val
                best_move = mv

            alpha = max(alpha, best_val)
            if beta <= alpha:
                break

        return best_val, best_move

    else:
        best_val = float("inf")
        best_move = None

        for mv in moves:
            child = apply_move(state, mv)
            val, _ = minmax(child, depth - 1, alpha, beta, maximizing_player, stats)

            if val < best_val:
                best_val = val
                best_move = mv

            beta = min(beta, best_val)
            if beta <= alpha:
                break

        return best_val, best_move
