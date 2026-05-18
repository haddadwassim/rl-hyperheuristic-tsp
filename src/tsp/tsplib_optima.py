TSPLIB_OPTIMA = {
    "eil51": 426,
    "berlin52": 7542,
    "st70": 675,
    "pr76": 108159,
    "rat99": 1211,
    "kroA100": 21282,
    "kroB100": 22141,
    "eil101": 629,
    "ch130": 6110,
}


def get_tsplib_optimum(instance_name: str) -> float | None:
    name = instance_name.replace(".tsp", "")
    return TSPLIB_OPTIMA.get(name)