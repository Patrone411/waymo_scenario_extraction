from worker_utils import (
    _encode_sparse_series,
    _encode_sparse_string_series,
    encode_inter_actor_pair,
)


def test_encode_sparse_series_with_gaps():
    result = _encode_sparse_series([None, 1, 2, None, 3])

    assert result == {
        "intervals": [[1, 2], [4, 4]],
        "data": [1, 2, 3],
    }


def test_encode_sparse_series_all_none():
    result = _encode_sparse_series([None, None, None])

    assert result == {
        "intervals": [],
        "data": [],
    }


def test_encode_sparse_series_without_gaps():
    result = _encode_sparse_series([1, 2, 3])

    assert result == {
        "intervals": [[0, 2]],
        "data": [1, 2, 3],
    }


def test_encode_sparse_string_series_ignores_invalid_values():
    result = _encode_sparse_string_series(
        [None, "unknown", "front", "front", "", "rear"]
    )

    assert result == {
        "intervals": [[2, 3], [5, 5]],
        "data": ["front", "front", "rear"],
    }


def test_encode_inter_actor_pair_empty_returns_none():
    result = encode_inter_actor_pair(
        {
            "ttc": [None, None],
            "eucl_distance": [None, None],
            "position": [None, "unknown", ""],
        }
    )

    assert result is None


def test_encode_inter_actor_pair():
    result = encode_inter_actor_pair(
        {
            "ttc": [None, 1.5, 2.0],
            "eucl_distance": [10.0, None, 8.0],
            "position": [None, "front", "front"],
        }
    )

    assert result == {
        "ttc": {
            "intervals": [[1, 2]],
            "data": [1.5, 2.0],
        },
        "eucl_distance": {
            "intervals": [[0, 0], [2, 2]],
            "data": [10.0, 8.0],
        },
        "position": {
            "intervals": [[1, 2]],
            "data": ["front", "front"],
        },
    }
