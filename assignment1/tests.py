# run tests using: pytest ./*/tests.py
import pytest
import numpy as np
from main import Shingling, CompareSets, MinHashing, LSH, CompareSignatures, reduce, map


@pytest.fixture
def documents():
    return ['This is a test', 'This is another test', 'This is a third test', 'Another test']


def test_shingling(documents):
    k = 5
    shingles = map(Shingling(k).shingle, documents)
    uniques = reduce(Shingling.intersection, shingles)
    assert len(shingles) == 4

    assert hash('This ') in uniques
    assert hash('This ') in shingles[0]
    assert hash('This ') in shingles[1]
    assert hash('This ') in shingles[2]
    assert hash('This ') not in shingles[3]

    chr_mtrx = Shingling.characteristic_matrix(shingles)
    assert chr_mtrx.shape == (len(uniques), 4)

    mapping_dict = {s: i for i, s in enumerate(uniques)}
    i = mapping_dict[hash('This ')]
    assert chr_mtrx[i, 0] == 1
    assert chr_mtrx[i, 1] == 1
    assert chr_mtrx[i, 2] == 1
    assert chr_mtrx[i, 3] == 0

    j = mapping_dict[hash(' test')]
    assert chr_mtrx[j, 0] == 1
    assert chr_mtrx[j, 1] == 1
    assert chr_mtrx[j, 2] == 1
    assert chr_mtrx[j, 3] == 1


def test_compare_sets():
    # Test Jaccard distance
    assert CompareSets.similarity({1, 2, 3}, {1, 2, 3}) == 1

    assert CompareSets.similarity({1, 2, 3}, {1, 2, 4}) == 2 / 4

    # Test Jaccard distance matrix
    matrix = CompareSets.similarity_matrix(
        [
            {1, 2, 3},
            {1, 2, 3, 4},
            {1, 2, 3, 4, 5},
        ]
    )

    expected_matrix = np.array(
        [
            [3 / 3, 3 / 4, 3 / 5],
            [3 / 4, 4 / 4, 4 / 5],
            [3 / 5, 4 / 5, 5 / 5],
        ]
    )

    assert np.allclose(matrix, expected_matrix), f'Error in distance matrix \n{matrix}\n{expected_matrix}'

    # Test getting similar document pairs
    pairs = CompareSets.similar_docs_from_mtrx(expected_matrix, threshold=0.5)

    assert pairs == {(0, 1), (0, 2), (1, 2)}

    pairs = CompareSets.similar_docs_from_mtrx(expected_matrix, threshold=0.6)

    assert pairs == {(0, 1), (1, 2)}

    pairs = CompareSets.similar_docs_from_mtrx(expected_matrix, threshold=0.75)

    assert pairs == {(1, 2)}
