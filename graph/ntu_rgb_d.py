import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]

# vnode
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                          (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                          (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                          (20, 19), (22, 23), (23, 8), (24, 25), (25, 12),

                            (26, 1),(26, 2),(26, 3),(26, 4),(26, 5),(26, 6),(26, 7),(26, 8),(26, 9),(26, 10),(26, 11),(26, 12),(26, 13),(26, 14),(26, 15),(26, 16),(26, 17),(26, 18),(26, 19),(26, 20),(26, 21),(26, 22),(26, 23),(26, 24),(26, 25)]

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
