from dataclasses import dataclass
from typing import List, Tuple

Move = Tuple[str, int, int]  # ("H"|"V", row, col)


@dataclass(frozen=True)
class GameState:
    n: int
    m: int
    h_lines: frozenset
    v_lines: frozenset
    box_owner: tuple
    scores: Tuple[int, int]
    player: int


def create_initial_state(n: int, m: int) -> GameState:
    if n < 2 or m < 2:
        raise ValueError("Board must be at least 2x2 points")

    return GameState(
        n=n,
        m=m,
        h_lines=frozenset(),
        v_lines=frozenset(),
        box_owner=tuple([-1] * ((n - 1) * (m - 1))),
        scores=(0, 0),
        player=0,
    )


def box_index(state: GameState, br: int, bc: int) -> int:
    return br * (state.m - 1) + bc


def box_sides_count(state: GameState, br: int, bc: int) -> int:
    top = (br, bc) in state.h_lines
    bottom = (br + 1, bc) in state.h_lines
    left = (br, bc) in state.v_lines
    right = (br, bc + 1) in state.v_lines
    return int(top) + int(bottom) + int(left) + int(right)


def is_box_completed(state: GameState, br: int, bc: int) -> bool:
    return box_sides_count(state, br, bc) == 4


def get_possible_moves(state: GameState) -> List[Move]:
    moves: List[Move] = []

    for r in range(state.n):
        for c in range(state.m - 1):
            if (r, c) not in state.h_lines:
                moves.append(("H", r, c))

    for r in range(state.n - 1):
        for c in range(state.m):
            if (r, c) not in state.v_lines:
                moves.append(("V", r, c))

    return moves


def _adjacent_boxes_for_move(state: GameState, move: Move) -> List[Tuple[int, int]]:
    line_type, r, c = move
    boxes: List[Tuple[int, int]] = []

    if line_type == "H":
        if r - 1 >= 0:
            boxes.append((r - 1, c))
        if r < state.n - 1:
            boxes.append((r, c))
    else:
        if c - 1 >= 0:
            boxes.append((r, c - 1))
        if c < state.m - 1:
            boxes.append((r, c))

    return boxes


def apply_move(state: GameState, move: Move) -> GameState:
    line_type, r, c = move
    new_h = set(state.h_lines)
    new_v = set(state.v_lines)

    if line_type == "H":
        if (r, c) in new_h:
            raise ValueError(f"Illegal move, line already drawn: {move}")
        new_h.add((r, c))
    elif line_type == "V":
        if (r, c) in new_v:
            raise ValueError(f"Illegal move, line already drawn: {move}")
        new_v.add((r, c))
    else:
        raise ValueError("line_type must be 'H' or 'V'")

    temp_state = GameState(
        n=state.n,
        m=state.m,
        h_lines=frozenset(new_h),
        v_lines=frozenset(new_v),
        box_owner=state.box_owner,
        scores=state.scores,
        player=state.player,
    )

    new_owner = list(state.box_owner)
    new_scores = list(state.scores)
    completed_now = 0

    for br, bc in _adjacent_boxes_for_move(temp_state, move):
        idx = box_index(temp_state, br, bc)
        if new_owner[idx] == -1 and is_box_completed(temp_state, br, bc):
            new_owner[idx] = state.player
            new_scores[state.player] += 1
            completed_now += 1

    next_player = state.player if completed_now > 0 else 1 - state.player

    return GameState(
        n=state.n,
        m=state.m,
        h_lines=frozenset(new_h),
        v_lines=frozenset(new_v),
        box_owner=tuple(new_owner),
        scores=(new_scores[0], new_scores[1]),
        player=next_player,
    )


def is_terminal(state: GameState) -> bool:
    total_lines = state.n * (state.m - 1) + (state.n - 1) * state.m
    return len(state.h_lines) + len(state.v_lines) == total_lines


def move_to_text(move: Move) -> str:
    line_type, r, c = move
    return f"{line_type}({r},{c})"


def render_board(state: GameState) -> str:
    rows: List[str] = []

    for r in range(state.n):
        top = []
        for c in range(state.m - 1):
            top.append("+")
            top.append("---" if (r, c) in state.h_lines else "   ")
        top.append("+")
        rows.append("".join(top))

        if r < state.n - 1:
            mid = []
            for c in range(state.m):
                mid.append("|" if (r, c) in state.v_lines else " ")
                if c < state.m - 1:
                    idx = box_index(state, r, c)
                    owner = state.box_owner[idx]
                    if owner == -1:
                        mid.append("   ")
                    else:
                        mid.append(f" {owner} ")
            rows.append("".join(mid))

    return "\n".join(rows)
