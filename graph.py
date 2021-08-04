class HeteroGraph():
    def __init__(self, nodes, edges, connects, node2idx) -> None:
        self.nodes = nodes
        self.edges = edges
        self.connects = connects
        self.node2idx = node2idx

