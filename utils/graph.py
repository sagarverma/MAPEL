from collections import defaultdict
import networkx as nx
import random

# This class represents a directed graph
# using adjacency list representation
class Graph:

    # Constructor
    def __init__(self):

        # default dictionary to store graph
        self.graph = defaultdict(list)

    def grid_to_graph(self, grid):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i][j] == 1:
                    pass
                else:
                    if i - 1 >= 0 and j - 1 >= 0 and grid[i-1][j-1] == 0:
                        self.addEdge((i,j), (i-1,j-1))
                    if j - 1 >= 0 and grid[i][j-1] == 0:
                        self.addEdge((i,j), (i,j-1))
                    if i + 1 < grid.shape[0] and j - 1 >= 0 and grid[i+1][j-1] == 0:
                        self.addEdge((i,j), (i+1,j-1))
                    if i - 1 >= 0 and grid[i-1][j] == 0:
                        self.addEdge((i,j), (i-1,j))
                    if i + 1 < grid.shape[0] and grid[i+1][j] == 0:
                        self.addEdge((i,j), (i+1,j))
                    if i - 1 >= 0 and j + 1 < grid.shape[1] and grid[i-1][j+1] == 0:
                        self.addEdge((i,j), (i-1,j+1))
                    if j + 1 < grid.shape[1] and grid[i][j+1] == 0:
                        self.addEdge((i,j), (i,j+1))
                    if i + 1 < grid.shape[0] and j + 1 < grid.shape[1] and grid[i+1][j+1] == 0:
                        self.addEdge((i,j), (i+1,j+1))

    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    # Function to print a BFS of graph
    def BFS(self, s):

        # Mark all the vertices as not visited
        visited = {}

        for g in self.graph.keys():
            visited[g] = False

        # Create a queue for BFS
        queue = []
        path = []

        # Mark the source node as
        # visited and enqueue it
        queue.append(s)
        visited[s] = True

        while queue:

            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0)
            path.append(s)

            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True

        return path

class NxGraph:
    # Constructor
    def __init__(self):

        # default dictionary to store graph
        self.graph = nx.Graph()

    def grid_to_graph(self, grid):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i][j] == 1:
                    pass
                else:
                    if i - 1 >= 0 and j - 1 >= 0 and grid[i-1][j-1] != 1:
                        self.graph.add_edge((i,j), (i-1,j-1))
                    if j - 1 >= 0 and grid[i][j-1] != 1:
                        self.graph.add_edge((i,j), (i,j-1))
                    if i + 1 < grid.shape[0] and j - 1 >= 0 and grid[i+1][j-1] != 1:
                        self.graph.add_edge((i,j), (i+1,j-1))
                    if i - 1 >= 0 and grid[i-1][j] != 1:
                        self.graph.add_edge((i,j), (i-1,j))
                    if i + 1 < grid.shape[0] and grid[i+1][j] != 1:
                        self.graph.add_edge((i,j), (i+1,j))
                    if i - 1 >= 0 and j + 1 < grid.shape[1] and grid[i-1][j+1] != 1:
                        self.graph.add_edge((i,j), (i-1,j+1))
                    if j + 1 < grid.shape[1] and grid[i][j+1] != 1:
                        self.graph.add_edge((i,j), (i,j+1))
                    if i + 1 < grid.shape[0] and j + 1 < grid.shape[1] and grid[i+1][j+1] != 1:
                        self.graph.add_edge((i,j), (i+1,j+1))

    def shortest_path(self, source, target):
        return nx.shortest_path(self.graph, source=source, target=target)

    def dijkstra_path(self, source, target):
        return nx.dijkstra_path(self.graph, source=source, target=target)

    def random_simple_path(self, source, target):
        simple_paths = list(nx.all_simple_paths(self.graph, source=source, target=target))
        return random.choice(simple_paths)