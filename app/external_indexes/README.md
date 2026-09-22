# Precomputed Indexes

Put already-built chunk-level indexes here when `USE_PRECOMPUTED_INDEXES=true`.

Required layout:

```text
external_indexes/
  bm25s_cache/
    chunk_ids.pkl              # or bm25s_chunk_ids.pkl / bm25_chunk_ids.pkl
    index/
      data.csc.index.npy
      indices.csc.index.npy
      indptr.csc.index.npy
      vocab.index.json
      params.index.json
  dense/
    dense_chunk_ids.pkl
    faiss_index.bin            # optional if chunk_embeddings.npy is present
    chunk_embeddings.npy       # fallback when faiss-cpu is not installed
  splade/
    splade_index.npz
    splade_chunk_ids.npy
```

Also provide the chunk mapping file:

```text
data/chunks_fixed_v1.parquet
```

It must contain:

- `chunk_id`
- `doc_id`
- `chunk_text` or `text` if snippets/reranking should use chunk text

The backend maps `chunk_id -> doc_id -> document metadata`, then deduplicates final results by
`doc_id` for doc-level metrics.
