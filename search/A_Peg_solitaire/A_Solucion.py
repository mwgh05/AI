import heapq
import time


# 1. Tablero y estado inicial

# El tablero inglés se modela como una matriz 7×7.


# Máscara de celdas válidas del tablero inglés
# Las celdas válidas forman una cruz; las esquinas son inválidas.
#   -1 = celda inválida (fuera de la cruz)
#    1 = celda con ficha (peg)
#    0 = celda vacía (hole)
VALID_MASK = [
    [-1, -1,  1,  1,  1, -1, -1],
    [-1, -1,  1,  1,  1, -1, -1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [-1, -1,  1,  1,  1, -1, -1],
    [-1, -1,  1,  1,  1, -1, -1],
]


VALID_POSITIONS = [(r, c) for r in range(7) for c in range(7) if VALID_MASK[r][c] != -1]


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def create_initial_board():
    board = [row[:] for row in VALID_MASK]
    board[3][3] = 0 
    return board


def board_to_tuple(board):
    """
    Convierte el tablero a tupla de tuplas.
    Esto permite usar el estado como clave en conjuntos y diccionarios
    """

    return tuple(tuple(row) for row in board)


def print_board(board):
    """
    Imprime el tablero de forma legible en consola.
    ● = ficha, ○ = vacío, espacio = celda inválida.
    """
    symbols = {-1: " ", 1: "●", 0: "○"}
    for row in board:
        print("  ".join(symbols[c] for c in row))
    print()


# 2. Movimientos y generación de sucesores

def get_valid_moves(board):
    """
    Genera todos los movimientos legales desde un estado dado.
    """
    moves = []
    for r, c in VALID_POSITIONS:
        if board[r][c] != 1:
            continue
        for dr, dc in DIRECTIONS:
            mr, mc = r + dr, c + dc        # celda intermedia (ficha saltada)
            tr, tc = r + 2 * dr, c + 2 * dc  # celda destino
            if 0 <= tr < 7 and 0 <= tc < 7:
                if board[mr][mc] == 1 and board[tr][tc] == 0:
                    moves.append((r, c, tr, tc))
    return moves


def apply_move(board, move):
    """
    Aplica un movimiento y retorna un nuevo estado del tablero.
    """
    r, c, tr, tc = move
    new_board = [row[:] for row in board]
    mr, mc = (r + tr) // 2, (c + tc) // 2  # posición intermedia
    new_board[r][c] = 0      # origen queda vacío
    new_board[mr][mc] = 0    # ficha saltada se elimina
    new_board[tr][tc] = 1    # destino recibe la ficha
    return new_board


# 3. Estados y verificaciones

def count_pegs(board):
    """
    Cuenta el número de fichas en el tablero.

    Args:
        board: Estado del tablero.

    Retorna:
        int: Cantidad de fichas presentes.
    """
    return sum(cell == 1 for row in board for cell in row)


def is_goal(board):
    """
    Verifica si se alcanzó el estado objetivo:
    una única ficha restante en el centro (3,3).

    Args:
        board: Estado del tablero.

    Retorna:
        bool: True si el estado es la meta.
    """
    return count_pegs(board) == 1 and board[3][3] == 1

# 4. heuristica principal

def heuristic(board):
    """
    Heurística admisible y consistente para Peg Solitaire.

    h(n) = (número de fichas restantes) - 1
    """
    return count_pegs(board) - 1


# ---------------------------------------------------------------------------
# 5. IMPLEMENTACIÓN DEL ALGORITMO A*
# ---------------------------------------------------------------------------

class AStarNode:
    """
    Atributos:
        board        : Estado actual del tablero (list[list[int]])
        g            : Costo acumulado desde el inicio (número de movimientos)
        h            : Valor heurístico estimado hasta la meta
        f            : f(n) = g(n) + h(n)
        parent       : Referencia al nodo padre (para reconstruir el camino)
        move         : Movimiento que generó este estado desde el padre
        _board_tuple : Representación hashable del tablero
    """
    __slots__ = ['board', 'g', 'h', 'f', 'parent', 'move', '_board_tuple']

    def __init__(self, board, g, parent=None, move=None):
        self.board = board
        self.g = g
        self.h = heuristic(board)
        self.f = self.g + self.h
        self.parent = parent
        self.move = move
        self._board_tuple = board_to_tuple(board)

    def __lt__(self, other):
        return self.f < other.f


def a_star_search(initial_board):
    """
    Ejecuta el algoritmo A* para resolver Peg Solitaire.
    """
    start_time = time.time()

    start_node = AStarNode(initial_board, g=0)

    open_list = []
    heapq.heappush(open_list, start_node)

    closed_set = set()

    # Contadores de estadísticas
    nodes_expanded = 0
    nodes_generated = 0

    while open_list:
        current = heapq.heappop(open_list)

        if current._board_tuple in closed_set:
            continue

        closed_set.add(current._board_tuple)
        nodes_expanded += 1

        if is_goal(current.board):
            elapsed = time.time() - start_time
            path = reconstruct_path(current)
            stats = {
                "tiempo_ejecucion_seg": round(elapsed, 4),
                "nodos_expandidos": nodes_expanded,
                "nodos_generados": nodes_generated,
                "movimientos": len(path) - 1,
                "estados_en_closed": len(closed_set),
            }
            return path, stats

        # expandir – generar sucesores
        for move in get_valid_moves(current.board):
            new_board = apply_move(current.board, move)
            new_tuple = board_to_tuple(new_board)

            if new_tuple not in closed_set:
                child = AStarNode(new_board, g=current.g + 1,
                                  parent=current, move=move)
                nodes_generated += 1
                heapq.heappush(open_list, child)

        # Reporte de progreso en caso de extensión masiva
        if nodes_expanded % 50_000 == 0:
            pegs = count_pegs(current.board)
            print(f"  [Progreso] Expandidos: {nodes_expanded:,} | "
                  f"Frontera: {len(open_list):,} | "
                  f"Fichas nodo actual: {pegs}")

    # Frontera vacía: no hay solución
    elapsed = time.time() - start_time
    stats = {
        "tiempo_ejecucion_seg": round(elapsed, 4),
        "nodos_expandidos": nodes_expanded,
        "nodos_generados": nodes_generated,
        "movimientos": -1,
        "estados_en_closed": len(closed_set),
    }
    return None, stats


def reconstruct_path(node):
    """
    Reconstruye el camino desde el nodo objetivo hasta el estado inicial.
    """
    path = []
    current = node
    while current is not None:
        path.append((current.move, current.board))
        current = current.parent
    path.reverse()
    return path


# 6. Pintado de movimientos legibles
def describe_move(move):
    if move is None:
        return "Estado inicial"
    r, c, tr, tc = move
    mr, mc = (r + tr) // 2, (c + tc) // 2
    return f"({r},{c}) → ({tr},{tc})  [salta sobre ({mr},{mc})]"


# 7. Busqueda con A* y ejecución principal
def main():
    print("=" * 65)
    print("   PEG SOLITAIRE – Solución con Algoritmo A*")
    print("=" * 65)
    print()

    board = create_initial_board()
    print("Tablero inicial:")
    print_board(board)
    print(f"Fichas iniciales: {count_pegs(board)}")
    print(f"Objetivo: 1 ficha en el centro (3,3)")
    print()

    print("Ejecutando A* ...")
    print("-" * 65)

    path, stats = a_star_search(board)

    print("-" * 65)
    print()

    if path is None:
        print("No se encontró solución.")
    else:
        print(f"¡Solución encontrada en {stats['movimientos']} movimientos!\n")

        for step_num, (move, board_state) in enumerate(path):
            print(f"--- Paso {step_num}: {describe_move(move)} ---")
            print_board(board_state)

    print("=" * 65)
    print("  ESTADÍSTICAS DE EJECUCIÓN")
    print("=" * 65)
    print(f"  Tiempo de ejecución   : {stats['tiempo_ejecucion_seg']} segundos")
    print(f"  Nodos expandidos      : {stats['nodos_expandidos']:,}")
    print(f"  Nodos generados       : {stats['nodos_generados']:,}")
    print(f"  Movimientos solución  : {stats['movimientos']}")
    print(f"  Estados en closed set : {stats['estados_en_closed']:,}")
    print("=" * 65)


if __name__ == "__main__":
    main()