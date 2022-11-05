# %% Imports
import numpy as np
from itertools import combinations
from collections import defaultdict
from nltk.corpus import reuters
from tqdm import tqdm

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
    '''Class that constructs k-shingles from a given document'''

    def __init__(self, k: int) -> None:
        self.k = k

    def shingle(self, text: str) -> set[int]:
        '''Constructs k-shingles from a given document, computes a hash value for each unique shingle and
        represents the document in the form of a set of its hashed k-shingles.'''
        shingles = [hash(text[i : i + self.k]) for i in range(len(text) - self.k + 1)]
        return set(shingles)

    @staticmethod
    def intersection(uniques: set[int], shingles: set[int]) -> set[int]:
        '''Computes the intersection of two sets of hashed k-shingles.'''
        return uniques | shingles


class CompareSets:
    @staticmethod
    def distance(set1: set[int], set2: set[int]) -> float:
        '''Computes the Jaccard distance between two sets'''
        return len(set1 & set2) / len(set1 | set2)

    @staticmethod
    def matrix(shingles: list[set[int]]) -> ndarray:
        '''Creates the Jaccard distance matrix for a given list of sets'''
        n = len(shingles)
        jaccard_matrix = np.zeros(shape=(n, n))
        for i, j in combinations(range(n), 2):
            jaccard_matrix[i, j] = CompareSets.distance(shingles[i], shingles[j])
            jaccard_matrix[j, i] = jaccard_matrix[i, j]
        return jaccard_matrix

    @staticmethod
    def similar_docs_from_mtrx(jaccard_matrix: ndarray, threshold: int) -> set[tuple[int, int]]:
        '''Returns a set of tuples of similar documents indices from a given Jaccard distance matrix'''
        n, _ = jaccard_matrix.shape
        similar_pairs = set()
        for doc1, doc2 in combinations(range(n), 2):
            if jaccard_matrix[doc1, doc2] > threshold:
                similar_pairs.add((doc1, doc2))
        return similar_pairs


class MinHashing:
    '''Class that builds a minHash signature of a given length n from a given set of integers'''

    def __init__(self, n: int) -> None:
        self.n_permutations = n

    def min_hash(self, char_mtrx: ndarray) -> ndarray:
        '''Builds a minHash signature of a given length n from a given set of integers'''
        n = self.n_permutations
        sign_mtrx = np.zeros(shape=(n, char_mtrx.shape[1]))
        for p in tqdm(range(n)):
            tmp_char_mtrx = np.random.permutation(char_mtrx)
            for c in range(tmp_char_mtrx.shape[1]):
                for r in range(tmp_char_mtrx.shape[0]):
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

    @staticmethod
    def test_candidates(candidate_pairs: set[tuple[int, int]], sign_mtrx: ndarray, threshold: float) -> set[tuple[int, int]]:
        similar_pairs = set()
        for i, j in candidate_pairs:
            sim = np.sum(sign_mtrx[:, i] == sign_mtrx[:, j]) / sign_mtrx.shape[0]
            if sim > threshold:
                similar_pairs.add((i, j))
        return similar_pairs

    @staticmethod
    def double_check(LSH_sim_pairs: set[tuple[int, int]], J_sim_pairs: set[tuple[int, int]]):
        TP = LSH_sim_pairs & J_sim_pairs
        FN = J_sim_pairs - LSH_sim_pairs
        FP = LSH_sim_pairs - J_sim_pairs
        return TP, FN, FP


class CompareSignatures:
    @staticmethod
    def approx_matrix(sign_mtrx: ndarray) -> ndarray:
        n_permutations, n_docs = sign_mtrx.shape
        j_mtrx = np.zeros(shape=(n_docs, n_docs))
        for i in range(n_docs):
            for j in range(n_docs):
                j_mtrx[i, j] = np.sum(sign_mtrx[:, i] == sign_mtrx[:, j]) / n_permutations
        return j_mtrx


# TODO: Find a better way to store the results of the various similarity methods and then plot them
def similar_documents_test(n_docs: list[int], ks: list[int], ss: list[float], n_permutations: list[int], n_bands: list[int]):
    for n in n_docs:
        docs = map(reuters.raw, reuters.fileids()[:n])
        for k in ks:
            shingling = Shingling(k)
            shingles = map(shingling.shingle, docs)
            j_mtrx = CompareSets.matrix(shingles)
            uniques = reduce(shingling.intersection, shingles)
            mapping_dict = {s: i for i, s in enumerate(uniques)}
            hashed_shingles = map(lambda l: map(mapping_dict.get, l), shingles)
            char_mtrx = np.zeros(shape=(len(uniques), len(docs)), dtype=int)
            for doc in range(len(hashed_shingles)):
                for int_shingle in hashed_shingles[doc]:
                    char_mtrx[int_shingle, doc] = 1
            for threshold in ss:
                sim_pairs = CompareSets.similar_docs_from_mtrx(j_mtrx, threshold=threshold)
                for n_perms in n_permutations:
                    sign_mtrx = MinHashing(n=n_perms).min_hash(char_mtrx=char_mtrx)
                    j_approx_mtrx = CompareSignatures.approx_matrix(sign_mtrx)
                    sim_pairs_approx = CompareSets.similar_docs_from_mtrx(j_approx_mtrx, threshold=threshold)
                    for b in n_bands:
                        lsh = LSH(n_bands=b)
                        candidate_pairs = lsh.get_candidate_pairs(sign_mtrx)
                        sim_pairs_LSH = lsh.test_candidates(candidate_pairs, sign_mtrx, threshold)


# %%
similar_documents_test(
    n_docs=[50], 
    ks=[2, 5, 10], 
    ss=[0.2], 
    n_permutations=[500], 
    n_bands=[1000]
    )

# TODO: implement a way to store how much time it takes

# %% Set parameters and load docs
k = 5
threshold = 0.2
n_permutations = 3000
b = 1000

n_docs = 50
docs = map(reuters.raw, reuters.fileids()[:n_docs])

# %% Get the shingles for each document
shingling = Shingling(k)
shingles = map(shingling.shingle, docs)

# %% Obtain all unique shingles and map each shingle to an int
uniques = reduce(shingling.intersection, shingles)
mapping_dict = {s: i for i, s in enumerate(uniques)}
hashed_shingles = map(lambda l: map(mapping_dict.get, l), shingles)

# %% Create characteristic mtrx (shingles x docs) and signature mtrx (signatures x docs)
char_mtrx = np.zeros(shape=(len(uniques), len(docs)), dtype=int)
for doc in range(len(hashed_shingles)):
    for int_shingle in hashed_shingles[doc]:
        char_mtrx[int_shingle, doc] = 1

sign_mtrx = MinHashing(n=n_permutations).min_hash(char_mtrx=char_mtrx)

# %% Obtain J matrx using all shingles and signatures
j_mtrx = CompareSets.matrix(shingles)
sim_pairs = CompareSets.similar_docs_from_mtrx(j_mtrx, threshold=threshold)

j_approx_mtrx = CompareSignatures.approx_matrix(sign_mtrx)
sim_pairs_approx = CompareSets.similar_docs_from_mtrx(j_approx_mtrx, threshold=threshold)

# %% LSH to find candidate pairs and test them
lsh = LSH(n_bands=b)
candidate_pairs = lsh.get_candidate_pairs(sign_mtrx)
sim_pairs_LSH = lsh.test_candidates(candidate_pairs, sign_mtrx, threshold)
TP, FN, FP = lsh.double_check(sim_pairs_LSH, sim_pairs)
# %%
print(len(candidate_pairs))
print(sum(map(len, shingles)))

# %%
