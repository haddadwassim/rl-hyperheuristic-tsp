from src.tsp.generators import generate_uniform_instance, generate_clustered_instance
from src.tsp.tour import nearest_neighbor_tour, random_tour, tour_length
from src.tsp.metrics import tour_edge_statistics, count_crossing_edges


def main():
    instance = generate_uniform_instance(num_nodes=20, seed=42)

    nn_tour = nearest_neighbor_tour(instance)
    rnd_tour = random_tour(instance)

    print("Uniform instance")
    print("Number of nodes:", instance.num_nodes)
    print("Nearest-neighbor length:", tour_length(nn_tour, instance))
    print("Random tour length:", tour_length(rnd_tour, instance))
    print("NN edge stats:", tour_edge_statistics(nn_tour, instance))
    print("NN crossing edges:", count_crossing_edges(nn_tour, instance))

    clustered = generate_clustered_instance(num_nodes=20, seed=42)
    clustered_tour = nearest_neighbor_tour(clustered)

    print("\nClustered instance")
    print("Number of nodes:", clustered.num_nodes)
    print("Nearest-neighbor length:", tour_length(clustered_tour, clustered))


if __name__ == "__main__":
    main()