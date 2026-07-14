import numpy as np
from scipy import sparse

from search.splade import SPLADEIndex


def test_splade_index_uses_saved_row_order(tmp_path):
    matrix = sparse.csr_matrix(
        [
            [0.0, 3.0],
            [5.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    row_indices = np.array([20, 10, 30], dtype=np.int64)
    sources = np.array(["a", "a", "b"])
    index = SPLADEIndex(matrix, row_indices, sources, {"fingerprint": ""})

    query = sparse.csr_matrix([[1.0, 0.0]], dtype=np.float32)
    hits = index.search(query, limit=2)

    assert [hit.row_index for hit in hits] == [10]


def test_splade_index_filters_sources_in_matrix_order():
    matrix = sparse.csr_matrix(
        [
            [0.0, 3.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    row_indices = np.array([2, 1], dtype=np.int64)
    sources = np.array(["keep", "drop"])
    index = SPLADEIndex(matrix, row_indices, sources, {"fingerprint": ""})

    query = sparse.csr_matrix([[0.0, 1.0]], dtype=np.float32)
    hits = index.search(query, limit=10, source_filter={"keep"})

    assert [hit.row_index for hit in hits] == [2]
