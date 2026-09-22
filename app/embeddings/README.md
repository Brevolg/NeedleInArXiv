# Embeddings directory

Place `embeddings_v1.npy` here. Its row order must be exactly the row order in
`data/id_mapping_v1.parquet`. The indexing script validates the row count,
dimension, dtype, and finite values before writing anything to Qdrant.

`embeddings_v2.npy` can be produced with `scripts/encode_corpus.py`.

