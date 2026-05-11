from tsp_hh.tsplib_loader import list_tsplib_files, load_tsplib_instance


def main():
    files = list_tsplib_files("data/tsplib")

    if not files:
        print("No .tsp files found in data/tsplib")
        print("Please place TSPLIB .tsp files there first.")
        return

    print("Found TSPLIB files:")

    for path in files:
        instance = load_tsplib_instance(path)

        print(
            f"{path.name:20s} | "
            f"n_cities={instance.coords.shape[0]:4d} | "
            f"distance_matrix={instance.distance_matrix.shape}"
        )


if __name__ == "__main__":
    main()