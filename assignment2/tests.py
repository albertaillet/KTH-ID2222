import pytest
from apriori import A_Priori


@pytest.fixture
def l1():
    # { b, c, j, m }, where b=1, c=2, m=3, j=4
    return {frozenset([1]), frozenset([2]), frozenset([3]), frozenset([4])}


@pytest.fixture
def l2():
    # { {b,m} {b,c} {c,m} {c,j} } where b=1, c=2, m=3, j=4
    return {frozenset([1, 3]), frozenset([1, 2]), frozenset([2, 3]), frozenset([2, 4])}


@pytest.fixture
def c3():
    # { {b,m,c} {b,c,j} {b,m,j} {c,m,j} } where b=1, c=2, m=3, j=4
    return {frozenset([1, 3, 2]), frozenset([1, 2, 4]), frozenset([1, 3, 4]), frozenset([2, 3, 4])}


@pytest.fixture
def baskets():
    # {A, C, D}, {B, C, E}, {A, B, C, E}, {B, E}, where A=1, B=2, C=3, D=4, E=5
    return [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]


def test_generate(l1, l2, c3):
    result = A_Priori.generate(l2, l1)
    assert result == c3


def test_prune(l2, c3):
    c3_pruned = {frozenset([1, 3, 2])}
    result = A_Priori.prune(c3, l2)
    assert result == c3_pruned


def test_count(baskets):
    expected_count_1 = {
        frozenset([1]): 2,
        frozenset([2]): 3,
        frozenset([3]): 3,
        frozenset([4]): 1,
        frozenset([5]): 3,
    }

    result = A_Priori.count(baskets, 1)

    assert result == expected_count_1

    expected_count_2 = {
        frozenset([1, 3]): 2,
        frozenset([1, 4]): 1,
        frozenset([2, 3]): 2,
        frozenset([2, 5]): 3,
        frozenset([3, 4]): 1,
        frozenset([3, 5]): 2,
    }

    candiates_2 = {
        frozenset([1, 3]),
        frozenset([1, 4]),
        frozenset([2, 3]),
        frozenset([2, 5]),
        frozenset([3, 4]),
        frozenset([3, 5]),
    }

    result = A_Priori.count(baskets, 2, candiates_2)

    assert result == expected_count_2

    expected_count_3 = {
        frozenset([1, 2, 3]): 1,
        frozenset([1, 3, 4]): 1,
        frozenset([1, 2, 5]): 1,
        frozenset([2, 3, 5]): 2,
    }

    candiates_3 = {
        frozenset([1, 2, 3]),
        frozenset([1, 3, 4]),
        frozenset([1, 2, 5]),
        frozenset([2, 3, 5]),
    }

    result = A_Priori.count(baskets, 3, candiates_3)

    assert result == expected_count_3


def test_filter():
    count = {
        frozenset([1, 3]): 2,
        frozenset([1, 4]): 4,
        frozenset([2, 3]): 2,
        frozenset([2, 5]): 6,
        frozenset([3, 4]): 1,
        frozenset([3, 5]): 2,
    }

    expected = {
        frozenset([1, 4]): 4,
        frozenset([2, 5]): 6,
    }
    filtered = A_Priori.filter(count, 3)
    assert filtered == expected

    expected = {
        frozenset([2, 5]): 6,
    }
    filtered = A_Priori.filter(count, 5)
    assert filtered == expected


def test_get_frequent_itemsets(baskets):
    results = A_Priori.get_frequent_itemsets(baskets, 3, 2)
    expected = {
        1: {
            frozenset([1]): 2,
            frozenset([2]): 3,
            frozenset([3]): 3,
            frozenset([5]): 3,
        },
        2: {
            frozenset([1, 3]): 2,
            frozenset([2, 3]): 2,
            frozenset([2, 5]): 3,
            frozenset([3, 5]): 2,
        },
        3: {
            frozenset([2, 3, 5]): 2,
        },
    }
    assert results == expected


def test_subsets():
    items = {1, 2, 3}
    expected = {
        frozenset([2, 3]),
        frozenset([1, 2]),
        frozenset([1, 3]),
        frozenset([2]),
        frozenset([3]),
        frozenset([1]),
    }
    result = A_Priori.subsets(items)
    assert result == expected

    items = {1, 2, 3, 4, 5}
    expected = {
        frozenset([1, 2, 3, 4]),
        frozenset([1, 2, 3, 5]),
        frozenset([1, 2, 4, 5]),
        frozenset([1, 3, 4, 5]),
        frozenset([2, 3, 4, 5]),
        frozenset([1, 2, 3]),
        frozenset([1, 2, 4]),
        frozenset([1, 2, 5]),
        frozenset([1, 3, 4]),
        frozenset([1, 3, 5]),
        frozenset([1, 4, 5]),
        frozenset([2, 3, 4]),
        frozenset([2, 3, 5]),
        frozenset([2, 4, 5]),
        frozenset([3, 4, 5]),
        frozenset([1, 2]),
        frozenset([1, 3]),
        frozenset([1, 4]),
        frozenset([1, 5]),
        frozenset([2, 3]),
        frozenset([2, 4]),
        frozenset([2, 5]),
        frozenset([3, 4]),
        frozenset([3, 5]),
        frozenset([4, 5]),
        frozenset([1]),
        frozenset([2]),
        frozenset([3]),
        frozenset([4]),
        frozenset([5]),
    }
    result = A_Priori.subsets(items)
    assert result == expected


def test_mine_frequent_rules():

    frequent_itemsets = {
        1: {
            frozenset([1]): 2,
            frozenset([2]): 3,
        },
        2: {
            frozenset([1, 2]): 2,
        },
    }

    expected = {
        frozenset([1, 2]): {
            frozenset([1]),
            frozenset([2]),
        }
    }
    result = A_Priori.mine_frequent_rules(frequent_itemsets, 2 / 3)
    assert result == expected

    expected = {
        frozenset([1, 2]): {
            frozenset([1]),
        }
    }
    result = A_Priori.mine_frequent_rules(frequent_itemsets, 1)
    assert result == expected

    frequent_itemsets = {
        1: {
            frozenset([1]): 2,
            frozenset([2]): 3,
            frozenset([3]): 3,
            frozenset([5]): 3,
        },
        2: {
            frozenset([1, 3]): 2,
            frozenset([2, 3]): 2,
            frozenset([2, 5]): 3,
            frozenset([3, 5]): 2,
        },
        3: {
            frozenset([2, 3, 5]): 2,
        },
    }

    expected = {
        frozenset([1, 3]): {frozenset([1])},
        frozenset([2, 5]): {frozenset([2]), frozenset([5])},
        frozenset([2, 3, 5]): {frozenset([2, 3]), frozenset([3, 5])},
    }

    result = A_Priori.mine_frequent_rules(frequent_itemsets, 1)

    assert result == expected
