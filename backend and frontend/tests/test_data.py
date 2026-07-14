from pathlib import Path

import pandas as pd
import pytest

from search.data import DataRepository


def test_repository_preserves_duplicate_rows(tmp_path: Path):
    path = tmp_path / "mapping.parquet"
    pd.DataFrame(
        {
            "doc_id": ["same", "same"],
            "source": ["jira", "slack"],
            "title": ["first", "second"],
            "char_len": [10, 20],
            "n_chunks": [1, 2],
        }
    ).to_parquet(path, index=False)
    repository = DataRepository(path)
    assert len(repository) == 2
    assert repository.duplicate_doc_ids == 2
    assert repository.document(1).title == "second"


def test_repository_rejects_missing_identifier(tmp_path: Path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"source": ["jira"], "title": ["x"]}).to_parquet(path, index=False)
    with pytest.raises(ValueError):
        DataRepository(path)

