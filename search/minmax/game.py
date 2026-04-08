import argparse

from runner import play_game_with_minmax, print_game_result, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dots and Boxes with Minmax + Alpha-Beta")
    parser.add_argument("--n", type=int, default=3, help="Rows of points")
    parser.add_argument("--m", type=int, default=3, help="Columns of points")
    parser.add_argument("--depth", type=int, default=5, help="Search depth for minmax")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark from depth 2..depth and try to plot",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result_data = play_game_with_minmax(n=args.n, m=args.m, depth=args.depth)
    print_game_result(result_data)

    if args.benchmark:
        run_benchmark(n=args.n, m=args.m, min_depth=2, max_depth=args.depth)

