import numpy as np

from utils import MinHeap

import Edge

COV_DIM = 6


class VertexGraph:

    def __init__(self, vertices_num, rel_covs=None, directed=True):
        # Todo: for now graph is for testing in Test.py
        """
        :param rel_covs: Relative covariances between consecutive cameras
        :param method: Graph's method for searching the shortest path.
            their is 3 options:
                1. Ajacency matrix
                2. Ajacency list with Min-Heap
                3. BFS
        :param edges : list of tuples (i, i + 1) such that there is an edge between 'i' and 'i + 1' vetrtices
        """

        self.__directed = directed
        self.__v_num = vertices_num
        self.__rel_covs = rel_covs
        self.create_vertex_graph()

    # === General code for creating graph and adding edges ===

    def create_vertex_graph(self):
        """
        Creates the vertex graph
        :return:
        """
        # Initialize the adjacency matrix and creates the graph
        self.__graph = [[Edge.Edge(row, col, None) for row in range(self.__v_num)] for col in range(self.__v_num)]

        if self.__rel_covs is not None:
            self.set_vertex_graph_adj_mat()

    def find_shortest_path(self, source, target):
        """
        Finds the shortest path  between the first_v vertex and target vertex according to the graph method and
        Returns it.
        """
        """
        Applys the Dijkstra algorithm for the adjacency matrix representation and finds the shortest path
        between first_v and target
        :return: Shortest path
        """
        dists = [float('inf')] * self.__v_num
        parents = [-1] * self.__v_num
        calculated_vertices = [False] * self.__v_num
        dists[source] = 0

        while calculated_vertices[target] is False:

            # Finds the vertex with the minimum distance from first_v and add it to the calculated vertices
            # at the first iteration it is the first_v itself
            min_dist_vertex = self.find_min_dist_vertex(calculated_vertices, dists)
            calculated_vertices[min_dist_vertex] = True

            # Go over all the vertices, v, and update the vertex's distance from the first_v by checking if
            # the distance from the first_v to it via the min_dist_vertex (first_v -> min_dist_vertex -> v)
            # is smaller than its current distance
            for v in range(self.__v_num):
                # First condition : If v is min_dist_vertex neighbor
                # Second condition : We want to update v's distance, so we check if it has been already calculated
                # Third condition : Check if the current distance is lower than the distance of the path that
                # pass through min_dist_vertex
                if self.__graph[min_dist_vertex][v].get_weight() > 0 \
                        and calculated_vertices[v] is False \
                        and dists[v] > dists[min_dist_vertex] + self.__graph[min_dist_vertex][v].get_weight():
                    # Update v's distance to the distance through the min_dist_vertex,
                    # and it's parents to the min_dist_vertex
                    dists[v] = dists[min_dist_vertex] + self.__graph[min_dist_vertex][v].get_weight()
                    parents[v] = min_dist_vertex

        # Compute path
        path = self.get_path(parents, target)

        return path

    def estimate_rel_cov(self, path):
        """
        Compute the estimated relative covariance between to cameras in the path the connecting them
        :param path: list of cameras indexes where the first index contains the first camera and the last index contains
         the last camera in the path
         :return estimated covariance
        """
        estimated_rel_cov = np.zeros((COV_DIM, COV_DIM))
        for i in range(1, len(path)):  # don't include first rel_covs at the path
            edge = self.get_edge_between_vertices(path[i - 1], path[i])
            estimated_rel_cov += edge.get_cov()
        return estimated_rel_cov

    def add_edge(self, source, target, cov):
        """
        Adds a directed edge of the form (first_v, target) with weight and covariance of cov
        """
        edge = Edge.Edge(source, target, cov)
        self.__graph[source][target] = edge

        if not self.__directed:
            edge = Edge.Edge(target, source, cov)
            self.__graph[target][source] = edge
    def get_edge_between_vertices(self, source, target):
        """
        Returns the edge between first_v and target
        """
        return self.__graph[source][target]



    def set_vertex_graph_adj_mat(self):
        """
       Creates the vertex graph - convert the basic pose graph to the vertex graph.
       The basic structure of the pose graph is as chain of cameras there for here we initialize
       only edges of consecutive vertex i and 'i+1'
       """
        for i in range(len(self.__rel_covs)):
            self.add_edge(i, i + 1, self.__rel_covs[i])

    def get_path(self, parents, target):
        """
        Recursive function for computing the path of target from the first_v
        :param parents: List of vertices parents
        :return: Path
        """
        # Base Case : If target is a first_v
        if parents[target] == -1:
            return [target]

        # Get recursively the path from first_v to target's parent
        return self.get_path(parents, parents[target]) + [target]

    def find_min_dist_vertex(self, calculated_vertices, dists):
        """
        Finds the vertex with the minimum distance from the first_v
        :param calculated_vertices: vertices that has been calculated already
        :param dists: List of vertices distances from the first_v
        :return: Vertex's index with the minimum distance between it and the first_v
        """
        minimum = float('inf')
        min_dist_vertex_ind = -1

        for u in range(len(dists)):
            if dists[u] < minimum and calculated_vertices[u] is False:
                minimum = dists[u]
                min_dist_vertex_ind = u

        return min_dist_vertex_ind
