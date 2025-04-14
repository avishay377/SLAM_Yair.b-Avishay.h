import numpy as np
import heapq

class Edge:
    def __init__(self, source, target, cov, weighted_det=True):
        """
        :param source: First vertex
        :param target: Second vertex
        :param cov: Covariance matrix
        :param weighted_det: If True, the edge is weighted; otherwise, it is not.
        """
        self.__source = source
        self.__target = target
        self.__cov = cov
        if weighted_det:
            self.__weight = np.linalg.det(cov)
        else:
            self.__weight = 1


class VertexGraph:




    def __init__(self, vertices_num, rel_covs=None, weighted_det = True):
        """
        :param rel_covs: Relative covariances between consecutive cameras
        :param directed: If True, the graph is directed; otherwise, it is undirected.
        """
        self.__v_num = vertices_num
        self.__rel_covs = rel_covs
        self.create_vertex_graph()

    def create_vertex_graph(self):
        """
        Creates the vertex graph as an adjacency list
        """
        self.__graph = {i: [] for i in range(self.__v_num)}
        for i in range(self.__v_num - 1):
            edge = Edge(i, i + 1, self.__rel_covs[i])
            self.__graph[i].append((i + 1, edge._Edge__weight))  # append tuple (target, weight)

    def find_shortest_path(self, source, target):
        """

        Args:
            source:
            target:

        Returns:

        """


        # use min heap for dists
        dists = [float('inf')] * self.__v_num

        parents = [-1] * self.__v_num
        calculated_vertices = [False] * self.__v_num
        dists[source] = 0
        min_heap = [(0, source)]  # (distance, node)

        while min_heap:
            curr_dist, u = heapq.heappop(min_heap)

            if calculated_vertices[u]:
                continue
            calculated_vertices[u] = True

            for neighbor, weight in self.__graph.get(u, []):
                if not calculated_vertices[neighbor] and dists[neighbor] > curr_dist + weight:
                    dists[neighbor] = curr_dist + weight
                    parents[neighbor] = u
                    heapq.heappush(min_heap, (dists[neighbor], neighbor))

        # Build the path from source to target (if needed)
        path = []
        if dists[target] != float('inf'):
            v = target
            while v != -1:
                path.append(v)
                v = parents[v]
            path.reverse()
        return dists[target], path



