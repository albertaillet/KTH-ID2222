# %% Imports
import numpy as np
from itertools import combinations
from collections import defaultdict
from nltk.corpus import reuters

# typing
from numpy import ndarray
from typing import Callable, Iterable, Any


def reduce(f: Callable, l: Iterable) -> Any:
    it = iter(l)
    value = next(it)
    for element in it:
        value = f(value, element)
    return value


def map(f: Callable, l: Iterable) -> list:
    return [f(x) for x in l]


class Shingling:
    def __init__(self, k: int) -> None:
        self.k = k

    def shingle(self, text: str) -> set[int]:
        shingles = [hash(text[i : i + self.k]) for i in range(len(text) - self.k + 1)]
        return set(shingles)

    @staticmethod
    def unique_shingles(uniques: set[int], shingles: set[int]) -> set[int]:
        return uniques | shingles


class CompareSets:
    @staticmethod
    def distance(set1: set[int], set2: set[int]) -> float:
        return len(set1 & set2) / len(set1 | set2)

    @staticmethod
    def matrix(shingles: list[set[int]]) -> ndarray:
        n_shingles = len(shingles)
        jaccard_matrix = np.zeros(shape=(n_shingles, n_shingles))
        for i in range(n_shingles):
            for j in range(n_shingles):
                jaccard_matrix[i, j] = CompareSets.distance(shingles[i], shingles[j])
        return jaccard_matrix


class MinHashing:
    def __init__(self, n_permutations: int) -> None:
        self.n_permutations = n_permutations

    def min_hash(self, char_mtrx: ndarray) -> ndarray:
        sign_mtrx = np.zeros(shape=(self.n_permutations, char_mtrx.shape[1]))
        n_unique_shingles = char_mtrx.shape[0]
        indices = np.arange(n_unique_shingles)
        for p in range(self.n_permutations):
            permutation = np.random.permutation(indices)
            tmp_char_mtrx = char_mtrx[permutation]
            for r in range(tmp_char_mtrx.shape[0]):
                for c in range(tmp_char_mtrx.shape[1]):
                    if tmp_char_mtrx[r, c]:
                        sign_mtrx[p, c] = r
                        break
        return sign_mtrx


class LSH:
    def __init__(self, n_bands: int) -> None:
        self.n_bands = n_bands

    def get_candidate_pairs(self, sign_mtrx: ndarray) -> set[tuple[int, int]]:
        buckets = defaultdict(set)
        n_bands = self.n_bands
        n_rows, n_docs = sign_mtrx.shape
        n_rows_per_band = n_rows // n_bands
        for band_idx in range(n_bands):
            start = band_idx * n_rows_per_band
            end = start + n_rows_per_band
            band = sign_mtrx[start:end]
            for i in range(n_docs):
                row = tuple(band[:, i])
                buckets[row].add(i)

        out = set()
        for bucket in buckets.values():
            if len(bucket) > 1:
                for doc1, doc2 in combinations(bucket, 2):
                    out.add((doc1, doc2))
        return out


class CompareSignatures:
    @staticmethod
    def approx_matrix(sign_mtrx: ndarray) -> ndarray:
        n_permutations, n_docs = sign_mtrx.shape
        j_mtrx = np.zeros(shape=(n_docs, n_docs))
        for i in range(n_docs):
            for j in range(n_docs):
                j_mtrx[i, j] = np.sum(sign_mtrx[:, i] == sign_mtrx[:, j]) / n_permutations
        return j_mtrx


# %% Load documents
n_docs = 5
docs = map(reuters.raw, reuters.fileids()[:n_docs])

# %% Get the shingles
shingling = Shingling(5)
shingles = map(shingling.shingle, docs)

# %%
uniques = reduce(shingling.unique_shingles, shingles)

# %%
distance_matrix = [[CompareSets.distance(s, t) for s in shingles] for t in shingles]

# %%
mapping_dict = {s: i for i, s in enumerate(uniques)}
hashed_shingles = map(lambda l: map(mapping_dict.get, l), shingles)

# %%
char_mtrx = np.zeros(shape=(len(uniques), len(docs)), dtype=int)
for d in range(len(hashed_shingles)):
    for int_shingle in hashed_shingles[d]:
        char_mtrx[int_shingle, d] = 1

sign_mtrx = MinHashing(n_permutations=1000).min_hash(char_mtrx=char_mtrx)

# %%
j_mtrx = CompareSets.matrix(shingles)
j_approx_mtrx = CompareSignatures.approx_matrix(sign_mtrx)
# %%
candidate_pairs = LSH(n_bands=500).get_candidate_pairs(sign_mtrx)

# %%
print(len(candidate_pairs))
print(sum(map(len, shingles)))

# %%
