from itertools import combinations
from typing import Optional


class A_Priori:
    
    @staticmethod
    def get_frequent_itemsets(baskets: list[set[int]], k_max: int, s: int) -> dict[int, set[frozenset[int]]]:
        '''Return frequent itemsets of size k_max for baskets with support s.'''
        l = {}

        counts_1 = A_Priori.count(baskets, 1)
        l[1] = A_Priori.filter(counts_1, s)
        for i in range(2, k_max+1):
            c_k = A_Priori.generate(l[i-1], l[1])
            pruned_c_k = A_Priori.prune(c_k, l[i-1])
            counts_i = A_Priori.count(baskets, i, pruned_c_k)
            l[i] = A_Priori.filter(counts_i, s)
        return l

    @staticmethod
    def count(baskets: list[set[int]], k: int, c_k: Optional[set[frozenset[int]]]=None) -> dict[frozenset[int], int]:
        '''Counts the number of occurances in baskets of each set for those that are also present in c_k.'''
        assert k == 1 or c_k is not None
        counts = {}
        for basket in baskets:
            for itemset in map(frozenset, combinations(basket, k)):
                if k == 1 or itemset in c_k: # type: ignore
                    counts[itemset] = counts.get(itemset, 0) + 1
        return counts
    
    @staticmethod
    def filter(counts: dict[frozenset[int], int], s: int) -> dict[frozenset[int], int]:
        '''Filters baskets to only include those that have a support above the threshold s'''
        return {itemset: count for itemset, count in counts.items() if count >= s}
    
    @staticmethod
    def generate(l_k_prev: set[frozenset[int]], l_1: dict[frozenset[int], int]) -> set[frozenset[int]]:
        '''Generates candidates in c_k can be  by combining itemsets from L_{k-1} and singletons from L_1.'''
        c_k = set()
        for itemset in l_k_prev:
            for singleton in l_1:
                candidate = itemset|singleton
                if len(candidate) == len(itemset) + 1:
                    c_k.add(candidate)
        return c_k

    @staticmethod 
    def prune(c_k: set[frozenset[int]], l_k_prev: set[frozenset[int]]) -> set[frozenset[int]]:
        '''Prunes candidates in c_k by checking if all subsets of size k-1 are in L_{k-1}.'''
        pruned_c_k = set()
        for itemset in c_k:
            k = len(itemset)
            inner_break = False
            for subset in combinations(tuple(itemset), k-1):
                if frozenset(subset) not in l_k_prev:
                    inner_break = True
                    break
            if not inner_break:
                pruned_c_k.add(itemset)
        return pruned_c_k  

    