import numpy as np
import pytest

from pseudolabel.retrieval import pair_positions, stratum_for_rank, top_k


def normed(rows):
    m = np.asarray(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_pair_positions_and_strata():
    a = normed([[1, 0]])
    b = normed([[1, 0], [0.9, 0.1], [0, 1]])
    ranks, cos = pair_positions(a, b, np.array([0, 0, 0]), np.array([0, 1, 2]))
    assert ranks.tolist() == [1, 2, 3]
    assert cos[0] == pytest.approx(1.0)
    assert stratum_for_rank(1) == "top_1_10" and stratum_for_rank(10) == "top_1_10"
    assert stratum_for_rank(11) == "top_11_100" and stratum_for_rank(100) == "top_11_100"
    assert stratum_for_rank(1000) == "top_101_1000" and stratum_for_rank(1001) is None


def test_pair_positions_several_queries():
    a = normed([[1, 0], [0, 1]])
    b = normed([[1, 0], [0, 1], [0.6, 0.8]])
    ranks, _ = pair_positions(a, b, np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert ranks.tolist() == [1, 1, 3]


def test_top_k_sorted():
    a = normed([[1, 0]])
    b = normed([[0, 1], [1, 0], [0.7, 0.7]])
    idx, cos = top_k(a, b, k=2)
    assert idx[0].tolist() == [1, 2] and cos[0, 0] >= cos[0, 1]


def test_top_k_larger_than_catalog():
    a = normed([[1, 0], [0, 1]])
    b = normed([[1, 0], [0, 1]])
    idx, _ = top_k(a, b, k=10, chunk=1)
    assert idx.shape == (2, 2) and idx[1, 0] == 1
