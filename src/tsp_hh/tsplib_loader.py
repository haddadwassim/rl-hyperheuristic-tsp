from pathlib import Path

import numpy as np

from tsp_hh.instances import TSPInstance, compute_distance_matrix


def parse_tsplib_coordinates(path: str | Path) -> np.ndarray:
    """
    Parse a TSPLIB .tsp file with NODE_COORD_SECTION.

    Supports common 2D Euclidean TSPLIB instances such as:
    - eil51
    - berlin52
    - st70
    - eil76
    - kroA100
    - ch130

    Returns
    -------
    coords:
        NumPy array of shape (n_cities, 2).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"TSPLIB file not found: {path}")

    coords = []
    in_coord_section = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.upper().startswith("NODE_COORD_SECTION"):
                in_coord_section = True
                continue

            if line.upper().startswith("EOF"):
                break

            if in_coord_section:
                parts = line.split()

                if len(parts) < 3:
                    continue

                # TSPLIB format: city_id x y
                _, x, y = parts[:3]

                coords.append([float(x), float(y)])

    if not coords:
        raise ValueError(f"No coordinates found in TSPLIB file: {path}")

    return np.asarray(coords, dtype=float)


def load_tsplib_instance(path: str | Path) -> TSPInstance:
    """
    Load a TSPLIB instance and convert it to the internal TSPInstance format.
    """
    coords = parse_tsplib_coordinates(path)
    distance_matrix = compute_distance_matrix(coords)

    return TSPInstance(
        coords=coords,
        distance_matrix=distance_matrix,
    )


def list_tsplib_files(data_dir: str | Path = "data/tsplib") -> list[Path]:
    """
    List available .tsp files in the TSPLIB data directory.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        return []

    return sorted(data_dir.glob("*.tsp"))