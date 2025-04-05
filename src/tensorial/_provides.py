import jraph


def determine_graph_batch_size(graphs: jraph.GraphsTuple):
    return (len(graphs.n_node) - jraph.get_number_of_padding_with_graphs_graphs(graphs),)


def get_batch_sizers() -> list:
    return [(jraph.GraphsTuple, determine_graph_batch_size)]
