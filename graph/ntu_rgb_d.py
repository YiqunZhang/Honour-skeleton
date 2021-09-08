import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]

inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12),

                    (1+25, 2+25), (2+25, 21+25), (3+25, 21+25), (4+25, 3+25), (5+25, 21+25), (6+25, 5+25), (7+25, 6+25),
                    (8+25, 7+25), (9+25, 21+25), (10+25, 9+25), (11+25, 10+25), (12+25, 11+25), (13+25, 1+25),
                    (14+25, 13+25), (15+25, 14+25), (16+25, 15+25), (17+25, 1+25), (18+25, 17+25), (19+25, 18+25),
                    (20+25, 19+25), (22+25, 23+25), (23+25, 8+25), (24+25, 25+25), (25+25, 12+25),

                    (1,1+25), (6,6+25), (16,16+25), (20,20+25), (25,25+25)

                    ]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
