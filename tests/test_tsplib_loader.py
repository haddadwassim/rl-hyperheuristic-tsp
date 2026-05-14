from pathlib import Path

from tsp_hh.tsplib_loader import parse_tsplib_coordinates, load_tsplib_instance


def test_parse_simple_tsplib_file(tmp_path: Path):
    tsp_content = """NAME: toy
TYPE: TSP
COMMENT: toy instance
DIMENSION: 4
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 1 0
3 1 1
4 0 1
EOF
"""

    path = tmp_path / "toy.tsp"
    path.write_text(tsp_content)

    coords = parse_tsplib_coordinates(path)

    assert coords.shape == (4, 2)
    assert coords[0].tolist() == [0.0, 0.0]
    assert coords[2].tolist() == [1.0, 1.0]


def test_load_simple_tsplib_instance(tmp_path: Path):
    tsp_content = """NAME: toy
TYPE: TSP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 1 0
3 1 1
4 0 1
EOF
"""

    path = tmp_path / "toy.tsp"
    path.write_text(tsp_content)

    instance = load_tsplib_instance(path)

    assert instance.coords.shape == (4, 2)
    assert instance.distance_matrix.shape == (4, 4)