# %% Imports
import numpy as np
from itertools import combinations
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from numba import njit
from time import time
import matplotlib.pyplot as plt

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
    @staticmethod
    def shingle(text: str, k: int) -> set[int]:
        '''Constructs k-shingles from a given document, computes a hash value for each unique shingle and
        represents the document in the form of a set of its hashed k-shingles.'''
        shingles = [hash(text[i : i + k]) for i in range(len(text) - k + 1)]
        return set(shingles)

    @staticmethod
    def intersection(uniques: set[int], shingles: set[int]) -> set[int]:
        '''Computes the intersection of two sets of hashed k-shingles.'''
        return uniques | shingles

    @staticmethod
    def characteristic_matrix(shingles: list[set[int]]) -> ndarray:
        '''Computes the characteristic matrix of a given set of documents.'''

        uniques = reduce(Shingling.intersection, shingles)
        mapping_dict = {s: i for i, s in enumerate(uniques)}
        hashed_shingles = map(lambda l: map(mapping_dict.get, l), shingles)

        n_unique_shingles, n_documents = len(uniques), len(shingles)
        char_mtrx = np.zeros(shape=(n_unique_shingles, n_documents), dtype=np.bool8)

        for doc in range(len(hashed_shingles)):
            for int_shingle in hashed_shingles[doc]:
                char_mtrx[int_shingle, doc] = True
        return char_mtrx


class CompareSets:
    @staticmethod
    def similarity(set1: set[int], set2: set[int]) -> float:
        '''Computes the Jaccard similarity between two sets'''
        return len(set1 & set2) / len(set1 | set2)

    @staticmethod
    def similarity_matrix(shingles: list[set[int]]) -> ndarray:
        '''Creates the Jaccard similarity matrix for a given list of sets'''
        n = len(shingles)
        jaccard_matrix = np.eye(n)
        for i, j in combinations(range(n), 2):
            jaccard_matrix[i, j] = CompareSets.similarity(shingles[i], shingles[j])
            jaccard_matrix[j, i] = jaccard_matrix[i, j]
        return jaccard_matrix

    @staticmethod
    def similar_docs_from_mtrx(jaccard_matrix: ndarray, threshold: float) -> set[tuple[int, int]]:
        '''Returns a set of tuples of similar documents indices from a given Jaccard distance matrix'''
        n, _ = jaccard_matrix.shape
        similar_pairs = set()
        for doc1, doc2 in combinations(range(n), 2):
            if jaccard_matrix[doc1, doc2] > threshold:
                similar_pairs.add((doc1, doc2))
        return similar_pairs


class MinHashing:
    @staticmethod
    @njit
    def hash(characteristic_matrix: ndarray, n: int, seed: int = 1) -> ndarray:
        '''Builds a minHash signature of a given length n from a given set of integers'''
        random_state = np.random.RandomState(seed)
        n_permutations = n
        n_unique_shingles, n_documents = characteristic_matrix.shape

        signature_mtrx = np.zeros((n_permutations, n_documents), dtype=np.int32)

        for permutation_index in tqdm(range(n_permutations)):
            permuted_characteristic_matrix = random_state.permutation(characteristic_matrix)
            for col_index in range(n_documents):
                row_index = 0
                while not permuted_characteristic_matrix[row_index, col_index]:
                    row_index += 1
                signature_mtrx[permutation_index, col_index] = row_index

        return signature_mtrx


class LSH:
    @staticmethod
    def get_candidate_pairs(signature_mtrx: ndarray, n_bands: int) -> set[tuple[int, int]]:
        '''Returns a set of candidate pairs of documents from a given signature matrix'''
        buckets = defaultdict(set)
        n_rows, n_docs = signature_mtrx.shape
        n_rows_per_band = n_rows // n_bands

        for band_idx in range(n_bands):
            start = band_idx * n_rows_per_band
            end = start + n_rows_per_band
            band = signature_mtrx[start:end]
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


def similar_documents_test(n_docs: list[int], ks: list[int], ss: list[float], n_permutations: list[int], n_bands: list[int]):
    sims_dict = {'J': {'sim_docs': [], 'time': []}, 'LSH': {'sim_docs': [], 'time': [], 'TP': []}, 'sign': {'sim_docs': [], 'time': [], 'TP': []}}
    for n in n_docs:
        docs = map(reuters.raw, reuters.fileids()[:n])
        for k in ks:
            shingles = map(partial(Shingling.shingle, k=k), docs)
            j_mtrx = CompareSets.similarity_matrix(shingles)
            char_mtrx = Shingling.characteristic_matrix(shingles)
            for threshold in ss:
                j_start_time = time()
                sim_pairs = CompareSets.similar_docs_from_mtrx(j_mtrx, threshold=threshold)
                j_time = time() - j_start_time

                sims_dict['J']['sim_docs'].append(len(sim_pairs))
                sims_dict['J']['time'].append(j_time)
                for n_perms in n_permutations:
                    sign_mtrx = MinHashing.hash(char_mtrx, n_perms)
                    j_approx_mtrx = CompareSignatures.approx_matrix(sign_mtrx)
                    sign_start_time = time()
                    sim_pairs_approx = CompareSets.similar_docs_from_mtrx(j_approx_mtrx, threshold=threshold)
                    sign_time = time() - sign_start_time
                    sims_dict['sign']['sim_docs'].append(len(sim_pairs_approx))
                    sims_dict['sign']['time'].append(sign_time)
                    sims_dict['sign']['TP'].append(len(sim_pairs & sim_pairs_approx))
                    for b in n_bands:
                        candidate_pairs = LSH.get_candidate_pairs(sign_mtrx, n_bands=b)
                        lsh_start_time = time()
                        sim_pairs_LSH = LSH.test_candidates(candidate_pairs, sign_mtrx, threshold)
                        lsh_time = time() - lsh_start_time
                        sims_dict['LSH']['sim_docs'].append(len(sim_pairs_LSH))
                        sims_dict['LSH']['time'].append(lsh_time)
                        sims_dict['LSH']['TP'].append(len(sim_pairs & sim_pairs_LSH))

    plt.figure(figsize=(10, 10))
    plt.suptitle(f'Similar Documents')  # k={k}, s={threshold}, n={n}, n_perms={n_perms}, n_bands={b}', fontsize=16)
    plt.subplot(2, 2, 1)
    plt.plot(n_docs, sims_dict['J']['sim_docs'], label='Jaccard')
    plt.plot(n_docs, sims_dict['sign']['sim_docs'], label='Signatures')
    plt.plot(n_docs, sims_dict['LSH']['sim_docs'], label='LSH')
    plt.xlabel('Number of documents')
    plt.ylabel('Number of similar documents')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(n_docs, sims_dict['J']['time'], label='Jaccard')
    plt.plot(n_docs, sims_dict['sign']['time'], label='Signatures')
    plt.plot(n_docs, sims_dict['LSH']['time'], label='LSH')
    plt.xlabel('Number of documents')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(n_docs, sims_dict['J']['sim_docs'], label='Jaccard')
    plt.plot(n_docs, sims_dict['sign']['TP'], label='Signatures')
    plt.plot(n_docs, sims_dict['LSH']['TP'], label='LSH')
    plt.xlabel('Number of documents')
    plt.ylabel('True positives')
    plt.legend()
    plt.show()


# %%
if __name__ == '__main__':
    from nltk.corpus import reuters

    similar_documents_test(n_docs=[50, 100, 200, 400], ks=[6], ss=[0.2], n_permutations=[1000], n_bands=[500])
