# imports
import random
from tqdm import tqdm
from collections import defaultdict
from abc import ABC, abstractmethod

# typing
from typing import Optional


class Trièst(ABC):
    '''Abstract class for the Trièst algorithm.'''

    def __init__(self, M: int, *, seed: int = 0) -> None:
        '''Initialize the Trièst algorithm.'''
        self.M: int = M
        self.S: list[tuple[int, int]] = [(0, 0)] * M  # initialize S with -1s
        self.t_global: float = 0.0
        self.t_local: defaultdict[int, float] = defaultdict(float)
        self.N = defaultdict(set)
        self.seed: int = seed
        random.seed(seed)

    @abstractmethod
    def __call__(self, stream: list[tuple[int, int]]) -> int:
        raise NotImplementedError

    @abstractmethod
    def update_counters(self, u: int, v: int, *, addition: bool = True, t: Optional[int] = None) -> None:
        raise NotImplementedError

    @staticmethod
    def reservoir_sampling(t: int, M: int):
        '''Returns -1 if the new edge is not kept in memory, otherwise the index of the reservoir that has to be substituted'''
        if t < M:
            return t
        elif Trièst.flip_biased_coin(M / t):
            return random.randrange(M)
        return -1

    @staticmethod
    def flip_biased_coin(p: float) -> bool:
        '''Returns True with probability p'''
        return random.random() < p


class TrièstBase(Trièst):
    def __call__(self, stream: list[tuple[int, int]]) -> int:
        '''Reads throught the edges sequentially, and updates the counters following the trièst-base algorithm.'''
        for t, (u, v) in tqdm(enumerate(stream)):
            rs = self.reservoir_sampling(t, self.M)
            if rs != -1:
                if t >= self.M:
                    (up, vp) = self.S[rs]
                    self.N[up].remove(vp)
                    self.N[vp].remove(up)
                    self.update_counters(up, vp, addition=False)

                self.S[rs] = (u, v)
                self.N[u].add(v)
                self.N[v].add(u)
                self.update_counters(u, v, addition=True)

        return int(self.t_global)

    def update_counters(self, u: int, v: int, *, addition: bool = True, t: Optional[int] = None) -> None:
        '''Updates the counters for estimating the number of triangles in S'''

        N_u_v = self.N[u] & self.N[v]

        for c in N_u_v:
            self.t_global += 1 if addition else -1
            self.t_local[u] += 1 if addition else -1
            self.t_local[v] += 1 if addition else -1
            self.t_local[c] += 1 if addition else -1


class TrièstImpr(Trièst):
    def __call__(self, stream: list[tuple[int, int]]) -> int:
        '''Reads throught the edges sequentially, and updates the counters following the trièst-impr algorithm.'''
        for t, (u, v) in tqdm(enumerate(stream)):
            self.update_counters(u, v, t=t)
            rs = self.reservoir_sampling(t, self.M)
            if rs != -1:
                if t >= self.M:
                    (up, vp) = self.S[rs]
                    self.N[up].remove(vp)
                    self.N[vp].remove(up)

                self.S[rs] = (u, v)
                self.N[u].add(v)
                self.N[v].add(u)

        return int(self.t_global)

    def update_counters(self, u: int, v: int, *, addition: bool = True, t: Optional[int] = None) -> None:
        '''Updates the counters for estimating the number of triangles in S'''

        N_u_v = self.N[u] & self.N[v]

        eta = max(1, ((t - 1) * (t - 2)) / (self.M * (self.M - 1)))

        for c in N_u_v:
            self.t_global += eta
            self.t_local[u] += eta
            self.t_local[v] += eta
            self.t_local[c] += eta
