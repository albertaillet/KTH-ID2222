
# imports
import numpy as np
import csv
from tqdm import tqdm
import networkx as nx

# typing
from collections import defaultdict
from typing import Union


class Trièst():
    def __init__(self, graph_name: str):
        if graph_name != '':
            self.edges = Trièst.load_data(graph_name)
            print(f'Data has been loaded!\nIt has {len(self.edges)} edges')
        

    @staticmethod
    def load_data(filename: str) -> list[tuple[int, int]]:
        '''Load data from file and return it as a list of frozensets.'''
        with open(f'data/{filename}/{filename}_edges.csv', newline='') as csvfile:
            edges_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            edges = list(tuple((int(a), int(b))) for a, b in list(edges_reader)[1:])
        return edges

    def reservoir_sampling(self, t: int) -> int:
        '''Returns -1 if the new edge is not kept in memory, otherwise the index of the reservoir that has to be substituted'''
        if t < self.M: 
            return t
        else:
            u = np.random.uniform()
            if u < self.M / t:                
                return np.random.choice(self.M)
            else:
                return -1

    def update_counters(self, edge: Union[tuple[int, int], tuple[None, None]], insert: bool=True, impr_t: int=0) -> None:
        '''Updates the counters for estimating the number of triangles in S'''
        u = edge[0]
        v = edge[1]

        N_u_v = self.N[u] & self.N[v]
        
        eta=1
        if impr_t:
            eta = max(1, ((impr_t - 1)*(impr_t - 2)) / (self.M*(self.M - 1)))

        for c in N_u_v:
            self.t_global += 1*eta if insert else -1*eta
            self.t_local[u] += 1*eta if insert else -1*eta  # type: ignore
            self.t_local[v] += 1*eta if insert else -1*eta  # type: ignore
            self.t_local[c] += 1*eta if insert else -1*eta

    def initialize(self, M):
        self.M = M
        self.S: list[Union[tuple[int, int], tuple[None, None]]] = [(None, None)] * M
        self.t_global: Union[int, float] = 0
        self.t_local: dict[int, Union[int, float]] = defaultdict(int) 
        self.N = defaultdict(set) 
        
    def trièst_base(self, M: int, T: int) -> tuple[Union[int, float], dict[int, Union[int, float]]]:
        '''Reads throught the edges sequentially, and updates the counters following the trièst-base algorithm.'''
        self.initialize(M)
        for t in tqdm(range(T)):
            rs = self.reservoir_sampling(t)
            if rs != -1:
                if t >= M:
                    edge_to_substitute = self.S[rs]
                    self.N[edge_to_substitute[0]].remove(edge_to_substitute[1])
                    self.N[edge_to_substitute[1]].remove(edge_to_substitute[0])
                    self.update_counters(edge_to_substitute, False)

                new_edge = self.edges[t]
                self.S[rs] = new_edge
                self.N[new_edge[0]].add(new_edge[1])
                self.N[new_edge[1]].add(new_edge[0])
                
                self.update_counters(new_edge)
        return self.t_global, self.t_local

    def trièst_impr(self, M: int, T: int) -> tuple[Union[int, float], dict[int, Union[int, float]]]:
        '''Reads throught the edges sequentially, and updates the counters following the trièst-impr algorithm.'''
        self.initialize(M)
        for t in tqdm(range(T)):
            self.update_counters(self.edges[t], impr_t=t)
            rs = self.reservoir_sampling(t)
            if rs != -1:
                if t >= M:
                    edge_to_substitute = self.S[rs]
                    self.N[edge_to_substitute[0]].remove(edge_to_substitute[1])
                    self.N[edge_to_substitute[1]].remove(edge_to_substitute[0])

                new_edge = self.edges[t]
                self.S[rs] = new_edge
                self.N[new_edge[0]].add(new_edge[1])
                self.N[new_edge[1]].add(new_edge[0])
        return int(self.t_global), self.t_local
                

    def triangles_count(self, T: int) -> int:
        '''generates a networkx graph using the first T edges and returns the number of triangles in it'''
        G = nx.Graph()
        G.add_edges_from(self.edges[:T])
        triangles = nx.triangles(G).values()
        return triangles
        