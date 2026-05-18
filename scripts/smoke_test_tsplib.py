import argparse
from pathlib import Path

from src.tsp.tsplib import load_tsplib_instance
from src.tsp.tour import nearest_neighbor_tour, tour_length


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test TSPLIB loader.")

    parser.add_argument(
        "--path",
        type=str,
        default="data/tsplib/berlin52.tsp",
        help="Path to TSPLIB .tsp file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    path = Path(args.path)

    instance = load_tsplib_instance(path)
    tour = nearest_neighbor_tour(instance)
    length = tour_length(tour, instance)

    print(f"Loaded: {path}")
    print(f"Number of nodes: {instance.num_nodes}")
    print(f"Nearest-neighbor tour length: {length:.3f}")


if __name__ == "__main__":
    main()