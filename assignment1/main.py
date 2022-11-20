# %% Imports
import numpy as np
from itertools import combinations
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import json

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
    def union(uniques: set[int], shingles: set[int]) -> set[int]:
        '''Computes the union of two sets of hashed k-shingles.'''
        return uniques | shingles

    @staticmethod
    def characteristic_matrix(shingles: list[set[int]]) -> ndarray:
        '''Computes the characteristic matrix of a given set of documents.'''

        uniques = reduce(Shingling.union, shingles)
        mapping_dict = {s: i for i, s in enumerate(uniques)}
        hashed_shingles = map(lambda l: map(mapping_dict.get, l), shingles)

        n_unique_shingles, n_documents = len(uniques), len(shingles)
        characteristic_matrix = np.zeros((n_unique_shingles, n_documents), dtype=np.bool8)

        for doc in range(len(hashed_shingles)):
            for int_shingle in hashed_shingles[doc]:
                characteristic_matrix[int_shingle, doc] = True
        return characteristic_matrix


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
    def threshold_similarity_matrix_pairs(jaccard_matrix: ndarray, threshold: float) -> set[tuple[int, int]]:
        '''Returns a set of tuples of similar documents indices from a given Jaccard similarity matrix'''
        n, _ = jaccard_matrix.shape
        similar_pairs = set()
        for doc1, doc2 in combinations(range(n), 2):
            if jaccard_matrix[doc1, doc2] > threshold:
                similar_pairs.add((doc1, doc2))
        return similar_pairs


class MinHashing:
    @staticmethod
    def hash(characteristic_matrix: ndarray, n: int, seed: int = 1) -> ndarray:
        '''Builds a minHash signature matrix of a given length n from a given set of integers'''
        random_state = np.random.RandomState(seed)
        n_permutations = n
        n_unique_shingles, n_documents = characteristic_matrix.shape

        signature_matrix = np.zeros((n_permutations, n_documents), dtype=np.int32)

        for permutation_index in tqdm(range(n_permutations)):
            permuted_characteristic_matrix = random_state.permutation(characteristic_matrix)
            for col_index in range(n_documents):
                row_index = 0
                while not permuted_characteristic_matrix[row_index, col_index]:
                    row_index += 1
                signature_matrix[permutation_index, col_index] = row_index

        return signature_matrix


class CompareSignatures:
    @staticmethod
    def similarity(v1: ndarray, v2: ndarray) -> Any:
        '''Computes the similarity between two minHash signatures'''
        return np.mean(v1 == v2)

    @staticmethod
    def approx_matrix(signature_mtrx: ndarray) -> ndarray:
        '''Computes the approximate Jaccard similarity matrix from a given signature matrix'''
        _, n_docs = signature_mtrx.shape
        approximate_similarity_matrix = np.eye(n_docs)
        for i, j in combinations(range(n_docs), 2):
            approximate_similarity_matrix[i, j] = CompareSignatures.similarity(signature_mtrx[:, i], signature_mtrx[:, j])
            approximate_similarity_matrix[j, i] = approximate_similarity_matrix[i, j]
        return approximate_similarity_matrix


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
    def test_candidates(candidate_pairs: set[tuple[int, int]], signature_mtrx: ndarray, threshold: float) -> set[tuple[int, int]]:
        '''Returns a set of similar pairs of documents from a given set of candidate pairs, thresholed by a given value'''
        n_rows, n_docs = signature_mtrx.shape
        similar_pairs = set()
        for i, j in candidate_pairs:
            sim = np.sum(signature_mtrx[:, i] == signature_mtrx[:, j]) / n_rows
            if sim > threshold:
                similar_pairs.add((i, j))
        return similar_pairs


def metrics(prediction: set[tuple[int, int]], ground_truth: set[tuple[int, int]], n: int) -> tuple[float, float, float]:
    n_total_pairs = n * (n - 1) // 2
    TPR = len(prediction & ground_truth) / len(ground_truth)
    FNR = len(ground_truth - prediction) / len(ground_truth)
    FPR = len(prediction - ground_truth) / (n_total_pairs - len(ground_truth))
    return TPR, FNR, FPR


def similar_documents_test(n_docs: list[int], k: int, threshold: float, n_permutations: int, n_bands: int):

    sims_dict = {
        'n_docs': [],
        'k': k,
        'threshold': threshold,
        'n_permutations': n_permutations,
        'n_bands': n_bands,
        'J': {'sim_docs': [], 'time': []},
        'LSH': {'sim_docs': [], 'time': [], 'TPR': [], 'FNR': [], 'FPR': []},
        'sign': {'sim_docs': [], 'time': [], 'TPR': [], 'FNR': [], 'FPR': []},
    }

    for n in n_docs:
        sims_dict['n_docs'].append(n)
        docs = map(reuters.raw, reuters.fileids()[:n])
        shingles = map(partial(Shingling.shingle, k=k), docs)

        j_start_time = time()
        j_mtrx = CompareSets.similarity_matrix(shingles)
        sim_pairs = CompareSets.threshold_similarity_matrix_pairs(j_mtrx, threshold=threshold)
        j_time = time() - j_start_time

        

        sims_dict['J']['sim_docs'].append(len(sim_pairs))
        sims_dict['J']['time'].append(j_time)

        char_mtrx = Shingling.characteristic_matrix(shingles)
        sign_mtrx = MinHashing.hash(char_mtrx, n_permutations)

        sign_start_time = time()
        j_approx_mtrx = CompareSignatures.approx_matrix(sign_mtrx)
        sim_pairs_approx = CompareSets.threshold_similarity_matrix_pairs(j_approx_mtrx, threshold=threshold)
        sign_time = time() - sign_start_time

        
        TPR, FNR, FPR = metrics(sim_pairs_approx, sim_pairs, n)

        sims_dict['sign']['sim_docs'].append(len(sim_pairs_approx))
        sims_dict['sign']['time'].append(sign_time)
        sims_dict['sign']['TPR'].append(TPR)
        sims_dict['sign']['FNR'].append(FNR)
        sims_dict['sign']['FPR'].append(FPR)
        candidate_pairs = LSH.get_candidate_pairs(sign_mtrx, n_bands)

        lsh_start_time = time()
        sim_pairs_LSH = LSH.test_candidates(candidate_pairs, sign_mtrx, threshold)
        lsh_time = time() - lsh_start_time

        TPR, FNR, FPR = metrics(sim_pairs_LSH, sim_pairs, n)

        sims_dict['LSH']['sim_docs'].append(len(sim_pairs_LSH))
        sims_dict['LSH']['time'].append(lsh_time)
        sims_dict['LSH']['TPR'].append(TPR)
        sims_dict['LSH']['FNR'].append(FNR)
        sims_dict['LSH']['FPR'].append(FPR)

        # save dict as json file
        with open('sims_dict.json', 'w') as f:
            json.dump(sims_dict, f, indent=4)


def plot_from_json(filename):
    with open(filename) as f:
        sims_dict = json.load(f)
    plt.figure(figsize=(15, 6))
    plt.suptitle(f'Similar Documents')
    plt.subplot(1, 2, 1)
    plt.title('Number of similar documents for each method')
    plt.plot(sims_dict['n_docs'], sims_dict['J']['sim_docs'], '-x', label='Jaccard')
    plt.plot(sims_dict['n_docs'], sims_dict['sign']['sim_docs'], '-x', label='Signatures')
    plt.plot(sims_dict['n_docs'], sims_dict['LSH']['sim_docs'], '-x', label='LSH')
    plt.xlabel('Number of documents')
    plt.ylabel('Number of similar documents')
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.title('Time to compute pairwise similarities')
    plt.plot(sims_dict['n_docs'], sims_dict['J']['time'], '-x', label='Jaccard')
    plt.plot(sims_dict['n_docs'], sims_dict['sign']['time'], '-x', label='Signatures')
    plt.plot(sims_dict['n_docs'], sims_dict['LSH']['time'], '-x', label='LSH')
    plt.xlabel('Number of documents')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid()
    # plt.subplot(2, 3, 3)
    # plt.title('True positive ratios')
    # plt.plot(sims_dict['n_docs'], sims_dict['sign']['TPR'], '-x', label='Signatures')
    # plt.plot(sims_dict['n_docs'], sims_dict['LSH']['TPR'], '-x', label='LSH')
    # plt.xlabel('Number of documents')
    # plt.ylabel('True positives')
    # plt.legend()
    # plt.subplot(2, 3, 4)
    # plt.title('False negative ratios')
    # plt.plot(sims_dict['n_docs'], sims_dict['sign']['FNR'], '-x', label='Signatures')
    # plt.plot(sims_dict['n_docs'], sims_dict['LSH']['FNR'], '-x', label='LSH')
    # plt.xlabel('Number of documents')
    # plt.ylabel('False negatives')
    # plt.legend()
    # plt.subplot(2, 3, 5)
    # plt.title('False positive ratios')
    # plt.plot(sims_dict['n_docs'], sims_dict['sign']['FPR'], '-x', label='Signatures')
    # plt.plot(sims_dict['n_docs'], sims_dict['LSH']['FPR'], '-x', label='LSH')
    # plt.xlabel('Number of documents')
    # plt.ylabel('False positives')
    # plt.legend()
    plt.show()


# %%
if __name__ == '__main__':
    from nltk.corpus import reuters

similar_documents_test(n_docs=[50, 100, 200, 400, 800], k=5, threshold=0.2, n_permutations=1000, n_bands=250)
plot_from_json('sims_dict.json')
# %%
docs = map(reuters.raw, reuters.fileids()[:1000])

# %%
shingles = map(partial(Shingling.shingle, k=5), docs)
# %%
