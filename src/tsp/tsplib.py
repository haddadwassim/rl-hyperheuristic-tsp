from pathlib import Path
import numpy as np

from src.tsp.instance import TSPInstance


def tsplib_distance_matrix(coordinates: np.ndarray, edge_weight_type: str) -> np.ndarray:
    """
    Compute TSPLIB-compatible distance matrix for common coordinate-based instances.
    """
    edge_weight_type = edge_weight_type.upper()

    diff = coordinates[:, None, :] - coordinates[None, :, :]
    euclidean = np.sqrt(np.sum(diff ** 2, axis=-1))

    if edge_weight_type == "EUC_2D":
        return np.floor(euclidean + 0.5).astype(np.float64)

    if edge_weight_type == "CEIL_2D":
        return np.ceil(euclidean).astype(np.float64)

    # ATT pseudo-Euclidean distance used by some TSPLIB instances.
    if edge_weight_type == "ATT":
        rij = np.sqrt(np.sum(diff ** 2, axis=-1) / 10.0)
        tij = np.floor(rij + 0.5)
        dij = np.where(tij < rij, tij + 1, tij)
        return dij.astype(np.float64)

    # Fallback: raw Euclidean distance.
    return euclidean.astype(np.float64)


def load_tsplib_instance(path: str | Path) -> TSPInstance:
    """
    Load a coordinate-based TSPLIB instance.

    Supported distance types:
    - EUC_2D
    - CEIL_2D
    - ATT

    For unsupported types, the loader falls back to raw Euclidean distances.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"TSPLIB file not found: {path}")

    coordinates = []
    in_coord_section = False
    edge_weight_type = "EUC_2D"

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            upper = line.upper()

            if upper.startswith("EDGE_WEIGHT_TYPE"):
                if ":" in line:
                    edge_weight_type = line.split(":", 1)[1].strip()
                else:
                    edge_weight_type = line.split()[-1].strip()

            if upper.startswith("NODE_COORD_SECTION"):
                in_coord_section = True
                continue

            if upper.startswith("EOF"):
                break

            if in_coord_section:
                parts = line.split()

                if len(parts) < 3:
                    continue

                x = float(parts[1])
                y = float(parts[2])
                coordinates.append([x, y])

    if not coordinates:
        raise ValueError(f"No coordinates found in TSPLIB file: {path}")

    coordinates = np.asarray(coordinates, dtype=np.float64)
    distance_matrix = tsplib_distance_matrix(coordinates, edge_weight_type)

    return TSPInstance(
        coordinates=coordinates,
        distance_matrix=distance_matrix,
    )